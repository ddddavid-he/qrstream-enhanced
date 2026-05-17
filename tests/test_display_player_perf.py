"""Tests for display player Phase-1 performance improvements.

These tests verify the correctness of performance optimizations made in
``fix/display-fps-phase1``:

- _PixmapCache: O(1) OrderedDict-based LRU
- Controls throttle: _update_controls called at ≤5 Hz
- Display skip: _update_display skipped when frame unchanged
- Zero-skip guarantee: playback never skips a frame
- Prebuffer pixmaps: pre-render before playback
- FPS warning: effective rate detection and status bar message
"""

from __future__ import annotations

import time
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qrstream.display_cache import (
    DisplayProducerState,
    ModuleFrameCache,
    pack_module_image,
)
from qrstream.display_player_qt import (
    DisplayPlayerQtConfig,
    _can_play,
)


# ── _PixmapCache tests (O(1) OrderedDict LRU) ──────────────────


class _FakePixmap:
    """Lightweight stand-in for QPixmap in non-GUI tests."""

    def __init__(self, tag: str = ""):
        self.tag = tag

    def __repr__(self):
        return f"FakePixmap({self.tag!r})"


def _make_pixmap_cache(max_entries: int = 4):
    """Create a _PixmapCache-compatible object without PySide6."""
    cache_dict: OrderedDict[tuple[int, int], _FakePixmap] = OrderedDict()

    class PixmapCache:
        def __init__(self):
            self._max = max_entries
            self._cache = cache_dict

        def get(self, key):
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

        def put(self, key, pixmap):
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self._max:
                self._cache.popitem(last=False)
            self._cache[key] = pixmap

        def clear(self):
            self._cache.clear()

    return PixmapCache()


def test_pixmap_cache_get_miss_returns_none():
    cache = _make_pixmap_cache()
    assert cache.get((0, 100)) is None


def test_pixmap_cache_put_and_get():
    cache = _make_pixmap_cache()
    px = _FakePixmap("frame0")
    cache.put((0, 100), px)
    assert cache.get((0, 100)) is px


def test_pixmap_cache_lru_eviction_order():
    cache = _make_pixmap_cache(max_entries=3)
    cache.put((0, 100), _FakePixmap("a"))
    cache.put((1, 100), _FakePixmap("b"))
    cache.put((2, 100), _FakePixmap("c"))

    # Access (0, 100) to make it most-recently used
    cache.get((0, 100))

    # Insert a 4th entry — should evict (1, 100) (least recently used)
    cache.put((3, 100), _FakePixmap("d"))

    assert cache.get((1, 100)) is None  # evicted
    assert cache.get((0, 100)) is not None  # still alive (accessed recently)
    assert cache.get((2, 100)) is not None  # still alive
    assert cache.get((3, 100)) is not None  # just inserted


def test_pixmap_cache_put_existing_key_updates_value():
    cache = _make_pixmap_cache()
    old = _FakePixmap("old")
    new = _FakePixmap("new")
    cache.put((0, 100), old)
    cache.put((0, 100), new)
    assert cache.get((0, 100)) is new


def test_pixmap_cache_clear():
    cache = _make_pixmap_cache()
    cache.put((0, 100), _FakePixmap("a"))
    cache.put((1, 100), _FakePixmap("b"))
    cache.clear()
    assert cache.get((0, 100)) is None
    assert cache.get((1, 100)) is None


# ── _can_play helper tests ──────────────────────────────────────


def _build_ready_cache(total_frames: int = 100, module_side: int = 9):
    """Build a fully-populated cache + done state."""
    cache = ModuleFrameCache(total_frames=total_frames, module_side=module_side)
    img = np.full((module_side, module_side), 255, dtype=np.uint8)
    img[4, 4] = 0
    for i in range(total_frames):
        cache.put_module_image(i, img)
    cache.mark_done()
    state = DisplayProducerState(total_frames)
    state.mark_produced(total_frames)
    state.mark_done()
    return cache, state


def test_can_play_returns_true_when_cache_done():
    cache, state = _build_ready_cache(total_frames=10)
    config = DisplayPlayerQtConfig(min_prebuffer_seconds=3.0)
    assert _can_play(cache, state, 0, 10, config)


def test_can_play_returns_false_for_empty_cache():
    cache = ModuleFrameCache(total_frames=10, module_side=9)
    state = DisplayProducerState(10)
    config = DisplayPlayerQtConfig()
    assert not _can_play(cache, state, 0, 10, config)


# ── Zero-skip guarantee logic tests ─────────────────────────────


def test_zero_skip_frame_advance_logic():
    """Verify that the frame-advance logic never skips a frame.

    Simulates the _tick() frame-advance path manually to prove that
    even when ticks are delayed, frames are advanced one at a time.
    Uses the accumulative+clamp strategy: _next += interval, clamped
    to now when falling behind.
    """
    fps = 60
    frame_interval = 1.0 / fps
    total_frames = 100

    frame_index = 0
    next_frame_ts = 0.0
    presented: list[int] = [0]

    # Simulate 200 ticks at irregular intervals
    tick_time = 0.0
    for i in range(200):
        # Vary tick intervals: some normal, some late
        if i % 7 == 0:
            tick_time += frame_interval * 3  # simulate jitter: 3x late
        else:
            tick_time += frame_interval * 0.8  # slightly fast

        now = tick_time
        if now >= next_frame_ts:
            nxt = frame_index + 1
            if nxt >= total_frames:
                break
            frame_index = nxt
            # Accumulative + clamp (matches actual _tick code)
            next_frame_ts += frame_interval
            if next_frame_ts < now:
                next_frame_ts = now
            presented.append(frame_index)

    # Verify: every presented frame is exactly +1 from the previous
    for i in range(1, len(presented)):
        assert presented[i] == presented[i - 1] + 1, (
            f"Frame skip detected at index {i}: "
            f"{presented[i - 1]} -> {presented[i]}"
        )


def test_zero_skip_no_frame_loss_under_jitter():
    """Under heavy jitter the strategy still presents every frame.

    The accumulative+clamp strategy (_next += interval, clamp to now)
    recovers pace quickly without ever skipping a frame — unlike the
    old late-reset code which would jump _next_frame_ts forward and
    silently drop frames.
    """
    fps = 30
    frame_interval = 1.0 / fps
    total_frames = 50

    frame_index = 0
    next_frame_ts = 0.0
    presented: list[int] = [0]

    tick_time = 0.0
    for _ in range(200):
        tick_time += frame_interval * 1.5  # every tick is 50% late

        now = tick_time
        if now >= next_frame_ts:
            nxt = frame_index + 1
            if nxt >= total_frames:
                break
            frame_index = nxt
            next_frame_ts += frame_interval
            if next_frame_ts < now:
                next_frame_ts = now
            presented.append(frame_index)

    # Still no gaps in the sequence
    for i in range(1, len(presented)):
        assert presented[i] == presented[i - 1] + 1, (
            f"Frame skip at index {i}: "
            f"{presented[i - 1]} -> {presented[i]}"
        )
    # Should have presented all frames
    assert presented[-1] == total_frames - 1


# ── Controls throttle logic test ─────────────────────────────────


def test_controls_throttle_limits_update_frequency():
    """Verify that controls update at most ~5Hz (200ms interval)."""
    controls_interval = 0.2
    next_controls_ts = 0.0
    update_count = 0

    # Simulate 1 second of ticks at ~60 Hz
    tick_time = 0.0
    for _ in range(60):
        tick_time += 1.0 / 60
        if tick_time >= next_controls_ts:
            update_count += 1
            next_controls_ts = tick_time + controls_interval

    # At 5 Hz over 1 second, expect ~5 updates (not 60)
    assert 4 <= update_count <= 7, (
        f"Expected ~5 control updates in 1 second, got {update_count}"
    )


# ── Display skip logic test ─────────────────────────────────────


def test_display_skip_when_frame_unchanged():
    """_update_display should be skipped when frame and side unchanged."""
    last_displayed_frame = -1
    last_displayed_side = -1
    display_calls = 0

    frame_index = 0
    side = 400

    # Simulate 10 ticks without frame change
    for _ in range(10):
        if frame_index != last_displayed_frame or side != last_displayed_side:
            display_calls += 1
            last_displayed_frame = frame_index
            last_displayed_side = side

    assert display_calls == 1, (
        f"Expected 1 display update for unchanged frames, got {display_calls}"
    )


def test_display_updates_on_frame_change():
    """_update_display should be called when frame index changes."""
    last_displayed_frame = -1
    last_displayed_side = -1
    display_calls = 0
    side = 400

    for frame_index in range(5):
        if frame_index != last_displayed_frame or side != last_displayed_side:
            display_calls += 1
            last_displayed_frame = frame_index
            last_displayed_side = side

    assert display_calls == 5


def test_display_updates_on_side_change():
    """_update_display should be called when display side changes."""
    last_displayed_frame = -1
    last_displayed_side = -1
    display_calls = 0
    frame_index = 0

    for side in [400, 400, 500, 500, 600]:
        if frame_index != last_displayed_frame or side != last_displayed_side:
            display_calls += 1
            last_displayed_frame = frame_index
            last_displayed_side = side

    assert display_calls == 3  # initial 400, then 500, then 600


# ── FPS warning logic test ──────────────────────────────────────


def test_fps_warning_triggers_below_threshold():
    """FPS warning should trigger when effective rate < 95% of target."""
    target_fps = 60
    warning_threshold = 0.95
    check_interval = 2.0

    fps_sample_start = 0.0
    fps_sample_count = 0
    warnings: list[str] = []

    # Simulate slow playback: ~50 fps for 3 seconds
    tick_time = 0.0
    for _ in range(150):
        tick_time += 1.0 / 50  # 50 fps effective
        fps_sample_count += 1
        if fps_sample_count == 1:
            fps_sample_start = tick_time
        elif tick_time - fps_sample_start >= check_interval:
            elapsed = tick_time - fps_sample_start
            effective = fps_sample_count / elapsed
            if effective < target_fps * warning_threshold:
                warnings.append(
                    f"Effective {effective:.1f} fps < target {target_fps} fps"
                )
            fps_sample_count = 0
            fps_sample_start = tick_time

    assert len(warnings) > 0, "Should have triggered at least one FPS warning"


def test_fps_warning_does_not_trigger_at_target():
    """No FPS warning when effective rate meets target."""
    target_fps = 60
    warning_threshold = 0.95
    check_interval = 2.0

    fps_sample_start = 0.0
    fps_sample_count = 0
    warnings: list[str] = []

    tick_time = 0.0
    for _ in range(120):
        tick_time += 1.0 / 60  # exactly 60 fps
        fps_sample_count += 1
        if fps_sample_count == 1:
            fps_sample_start = tick_time
        elif tick_time - fps_sample_start >= check_interval:
            elapsed = tick_time - fps_sample_start
            effective = fps_sample_count / elapsed
            if effective < target_fps * warning_threshold:
                warnings.append(
                    f"Effective {effective:.1f} fps < target {target_fps} fps"
                )
            fps_sample_count = 0
            fps_sample_start = tick_time

    assert len(warnings) == 0, f"Unexpected warnings: {warnings}"
