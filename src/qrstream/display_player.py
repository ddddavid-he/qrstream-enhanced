"""OpenCV playback loop for display-only encoding."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Event, RLock
import time

from ._compat import suppress_native_stderr
from .display_cache import (
    DEFAULT_PRESENTATION_CACHE_BUDGET,
    ModuleFrameCache,
    PresentationFrameCache,
)

with suppress_native_stderr():
    import cv2
    import numpy as np


@dataclass(frozen=True)
class DisplayPlayerConfig:
    window_name: str = "QRStream Display"
    initial_module_px: int = 8
    max_display_side: int = 1200
    min_prebuffer_seconds: float = 3.0
    producer_fps_window_seconds: float = 3.0
    producer_grace_factor: float = 1.05
    presentation_cache_budget: int = DEFAULT_PRESENTATION_CACHE_BUDGET


class DisplayProducerState:
    """Thread-safe producer progress shared with the player."""

    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        self._lock = RLock()
        self._done = Event()
        self._cancel = Event()
        self._started = time.monotonic()
        self._produced = 0
        self._samples: deque[tuple[float, int]] = deque()

    def mark_produced(self, count: int = 1) -> None:
        now = time.monotonic()
        with self._lock:
            self._produced += count
            self._samples.append((now, self._produced))
            self._trim_samples(now, 10.0)

    def mark_done(self) -> None:
        self._done.set()

    def request_cancel(self) -> None:
        self._cancel.set()

    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def is_done(self) -> bool:
        return self._done.is_set()

    def wait_done(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    @property
    def produced(self) -> int:
        with self._lock:
            return self._produced

    @property
    def progress_pct(self) -> float:
        if self.total_frames <= 0:
            return 100.0
        return min(100.0, self.produced / self.total_frames * 100.0)

    def producer_fps(self, window_seconds: float = 3.0) -> float:
        now = time.monotonic()
        with self._lock:
            self._trim_samples(now, max(window_seconds, 0.1))
            if len(self._samples) >= 2:
                first_ts, first_count = self._samples[0]
                last_ts, last_count = self._samples[-1]
                elapsed = max(1e-6, last_ts - first_ts)
                return max(0.0, (last_count - first_count) / elapsed)
            elapsed = max(1e-6, now - self._started)
            return self._produced / elapsed

    def _trim_samples(self, now: float, window_seconds: float) -> None:
        cutoff = now - window_seconds
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.popleft()


def _display_side(module_side: int, module_px: int, max_side: int) -> int:
    px = max(1, int(module_px))
    if max_side > 0:
        px = max(1, min(px, max_side // module_side))
    return module_side * px


def _can_play(cache: ModuleFrameCache, state: DisplayProducerState,
              frame_index: int, fps: int, config: DisplayPlayerConfig) -> bool:
    contiguous = cache.contiguous_from(frame_index)
    if contiguous <= 0:
        return False
    if cache.is_done() or state.is_done():
        return True
    min_frames = max(1, int(config.min_prebuffer_seconds * max(1, fps)))
    if contiguous < min_frames:
        return False
    producer_fps = state.producer_fps(config.producer_fps_window_seconds)
    return producer_fps >= fps * config.producer_grace_factor


def _render_frame(cache: ModuleFrameCache, presentation: PresentationFrameCache,
                  frame_index: int, display_side: int) -> np.ndarray | None:
    key = (frame_index, display_side)
    cached = presentation.get(key)
    if cached is not None:
        return cached
    module_img = cache.get_module_image(frame_index)
    if module_img is None:
        return None
    scaled = cv2.resize(
        module_img,
        (display_side, display_side),
        interpolation=cv2.INTER_NEAREST,
    )
    frame = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    presentation.put(key, frame)
    return frame.copy()


def _placeholder(display_side: int) -> np.ndarray:
    return np.full((display_side, display_side, 3), 255, dtype=np.uint8)


_STATUS_PANEL_HEIGHT = 96
_STEP_BACK_KEYS = {ord('a'), ord('A'), ord(','), 81, 2424832, 65361, 63234}
_STEP_FORWARD_KEYS = {ord('d'), ord('D'), ord('.'), 83, 2555904, 65363, 63235}
_JUMP_BACK_KEYS = {ord('j'), ord('J'), ord('['), 84, 2621440, 65364, 63233}
_JUMP_FORWARD_KEYS = {ord('l'), ord('L'), ord(']'), 82, 2490368, 65362, 63232}


def _window_height(display_side: int) -> int:
    return display_side + _STATUS_PANEL_HEIGHT


def _compose_status_frame(qr_frame: np.ndarray, *, frame_index: int,
                          cache: ModuleFrameCache,
                          state: DisplayProducerState, fps: int,
                          playing: bool, can_play: bool) -> np.ndarray:
    """Return a display canvas with status text outside the QR image."""
    total = max(1, cache.total_frames)
    cached_pct = cache.valid_count / total * 100.0
    producer_fps = state.producer_fps()
    status = "PLAY" if playing else "PAUSE"
    if not can_play and not cache.is_done():
        status = "BUFFERING"

    h, w = qr_frame.shape[:2]
    canvas = np.full((h + _STATUS_PANEL_HEIGHT, w, 3), 255, dtype=np.uint8)
    canvas[:h, :w] = qr_frame
    cv2.line(canvas, (0, h), (w, h), (220, 220, 220), 1)

    lines = [
        f"{status}  frame {min(frame_index + 1, total)}/{cache.total_frames}  "
        f"cache {cached_pct:5.1f}%  producer {producer_fps:5.1f} fps  target {fps} fps",
        "SPACE play/pause | Left/Right or A/D frame | J/L or Down/Up +/-1s",
        "+/- scale | Q/ESC quit",
    ]
    y = h + 24
    for text in lines:
        cv2.putText(
            canvas,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (32, 32, 32),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return canvas


def _seek_if_cached(cache: ModuleFrameCache, current_index: int,
                    target_index: int) -> int:
    target = max(0, min(cache.total_frames - 1, target_index))
    if target == current_index:
        return current_index
    if cache.has_frame(target):
        return target
    return current_index


def play_display_cache(cache: ModuleFrameCache, state: DisplayProducerState,
                       fps: int,
                       config: DisplayPlayerConfig | None = None) -> None:
    """Play cached module frames in an OpenCV window.

    The window opens paused. Playback starts only after the user presses
    space and the cache/producer-speed gate says playback can keep up.
    """
    if config is None:
        config = DisplayPlayerConfig()

    module_px = max(1, config.initial_module_px)
    display_side = _display_side(
        cache.module_side, module_px, config.max_display_side)
    presentation = PresentationFrameCache(config.presentation_cache_budget)
    frame_index = 0
    playing = False
    frame_interval = 1.0 / fps if fps > 0 else 0.1
    next_frame_ts = time.monotonic()

    try:
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.window_name, display_side, _window_height(display_side))
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV display window is unavailable. Install a GUI-enabled "
            "OpenCV build and run in a desktop session to use --display."
        ) from exc

    try:
        while True:
            can_play = _can_play(cache, state, frame_index, fps, config)
            if playing and not can_play:
                playing = False

            frame = _render_frame(cache, presentation, frame_index, display_side)
            if frame is None:
                frame = _placeholder(display_side)
            canvas = _compose_status_frame(
                frame,
                frame_index=frame_index,
                cache=cache,
                state=state,
                fps=fps,
                playing=playing,
                can_play=can_play,
            )

            try:
                cv2.imshow(config.window_name, canvas)
            except cv2.error as exc:
                raise RuntimeError(
                    "OpenCV display window is unavailable. Install a GUI-enabled "
                    "OpenCV build and run in a desktop session to use --display."
                ) from exc

            now = time.monotonic()
            delay_ms = 30
            if playing:
                delay_ms = max(1, min(30, int((next_frame_ts - now) * 1000)))
            wait_key = getattr(cv2, 'waitKeyEx', cv2.waitKey)
            key = wait_key(delay_ms)
            key_ascii = key & 0xFF

            if key_ascii in (27, ord('q'), ord('Q')):
                state.request_cancel()
                break
            if key_ascii == ord(' '):
                if playing:
                    playing = False
                elif can_play:
                    playing = True
                    next_frame_ts = time.monotonic() + frame_interval
            elif key in _STEP_BACK_KEYS or key_ascii in _STEP_BACK_KEYS:
                frame_index = _seek_if_cached(cache, frame_index, frame_index - 1)
                playing = False
            elif key in _STEP_FORWARD_KEYS or key_ascii in _STEP_FORWARD_KEYS:
                frame_index = _seek_if_cached(cache, frame_index, frame_index + 1)
                playing = False
            elif key in _JUMP_BACK_KEYS or key_ascii in _JUMP_BACK_KEYS:
                frame_index = _seek_if_cached(cache, frame_index, frame_index - max(1, fps))
                playing = False
            elif key in _JUMP_FORWARD_KEYS or key_ascii in _JUMP_FORWARD_KEYS:
                frame_index = _seek_if_cached(cache, frame_index, frame_index + max(1, fps))
                playing = False
            elif key_ascii in (ord('+'), ord('=')):
                module_px += 1
                display_side = _display_side(
                    cache.module_side, module_px, config.max_display_side)
                presentation.clear()
                cv2.resizeWindow(config.window_name, display_side, _window_height(display_side))
            elif key_ascii in (ord('-'), ord('_')) and module_px > 1:
                module_px -= 1
                display_side = _display_side(
                    cache.module_side, module_px, config.max_display_side)
                presentation.clear()
                cv2.resizeWindow(config.window_name, display_side, _window_height(display_side))

            if playing and time.monotonic() >= next_frame_ts:
                if frame_index + 1 >= cache.total_frames:
                    playing = False
                elif cache.has_frame(frame_index + 1):
                    frame_index += 1
                    next_frame_ts += frame_interval
                    if next_frame_ts < time.monotonic() - frame_interval:
                        next_frame_ts = time.monotonic() + frame_interval
                else:
                    playing = False

            if state.cancel_requested():
                break
    finally:
        try:
            cv2.destroyWindow(config.window_name)
        except cv2.error:
            pass
