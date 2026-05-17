"""Tests for display-only frame caches."""

import numpy as np

from qrstream.display_cache import (
    DisplayProducerState,
    ModuleFrameCache,
    SharedBufferCacheAdapter,
    SharedFrameBuffer,
    SharedProducerState,
    estimate_module_cache_bytes,
    pack_module_image,
    plan_module_cache,
    unpack_module_frame,
)


def test_pack_unpack_module_image_roundtrip():
    module_img = np.full((9, 9), 255, dtype=np.uint8)
    module_img[1, 2] = 0
    module_img[3, 8] = 0

    packed = pack_module_image(module_img)
    assert packed.shape == (9, 2)

    unpacked = unpack_module_frame(packed, module_side=9)
    assert np.array_equal(unpacked, module_img)


def test_module_frame_cache_tracks_contiguous_frames():
    cache = ModuleFrameCache(total_frames=5, module_side=9, chunk_size=2)
    img = np.full((9, 9), 255, dtype=np.uint8)
    img[4, 4] = 0

    cache.put_module_image(0, img)
    cache.put_module_image(1, img)
    cache.put_module_image(3, img)

    assert cache.valid_count == 3
    assert cache.contiguous_from(0) == 2
    assert cache.contiguous_from(2) == 0
    assert cache.contiguous_from(3) == 1
    assert np.array_equal(cache.get_module_image(1), img)


def test_module_cache_plan_defaults_to_full_for_default_one_hour_shape():
    total_frames = 3600 * 10
    module_side = 125

    assert estimate_module_cache_bytes(total_frames, module_side) == 72_000_000
    plan = plan_module_cache(total_frames, module_side, fps=10)
    assert plan.mode == "full"
    assert plan.total_bytes == 72_000_000


def test_module_cache_plan_uses_window_above_thresholds():
    plan = plan_module_cache(total_frames=200_000, module_side=177, fps=10)
    assert plan.mode == "window"
    assert plan.memory_budget_bytes < plan.total_bytes


def test_display_producer_state_tracks_progress_and_completion():
    state = DisplayProducerState(total_frames=10)

    state.mark_produced(3)
    assert state.produced == 3
    assert state.progress_pct == 30.0
    assert state.producer_fps() >= 0.0
    assert not state.is_done()

    state.mark_done()
    assert state.is_done()
    assert state.wait_done(timeout=0.01)


def test_display_producer_state_cancel_flag_roundtrip():
    state = DisplayProducerState(total_frames=1)
    assert not state.cancel_requested()
    state.request_cancel()
    assert state.cancel_requested()


# ── SharedFrameBuffer tests ─────────────────────────────────────


def test_shared_frame_buffer_put_get_roundtrip():
    buf = SharedFrameBuffer(total_frames=5, module_side=9)
    try:
        img = np.full((9, 9), 255, dtype=np.uint8)
        img[4, 4] = 0
        packed = pack_module_image(img)

        buf.put_packed(0, packed)
        buf.put_packed(3, packed)

        assert buf.has_frame(0)
        assert not buf.has_frame(1)
        assert buf.has_frame(3)
        assert not buf.has_frame(5)   # out of range
        assert not buf.has_frame(-1)  # negative

        got = buf.get_packed(0)
        assert got is not None
        assert np.array_equal(got, packed.reshape(9, (9 + 7) // 8))

        assert buf.get_packed(1) is None
        assert buf.get_packed(-1) is None
        assert buf.get_packed(5) is None
    finally:
        buf.close()


def test_shared_frame_buffer_unpack_matches_original():
    buf = SharedFrameBuffer(total_frames=3, module_side=21)
    try:
        rng = np.random.default_rng(42)
        for i in range(3):
            img = np.where(
                rng.random((21, 21)) > 0.5, 0, 255
            ).astype(np.uint8)
            packed = pack_module_image(img)
            buf.put_packed(i, packed)

            got = buf.get_packed(i)
            unpacked = unpack_module_frame(got, 21)
            assert np.array_equal(unpacked, img), f"Frame {i} mismatch"
    finally:
        buf.close()


# ── SharedProducerState tests ───────────────────────────────────


def test_shared_producer_state_produced_and_done():
    state = SharedProducerState(total_frames=10)
    assert state.produced == 0
    assert not state.is_done()

    state.mark_produced(3)
    assert state.produced == 3
    assert state.progress_pct == 30.0

    state.mark_done()
    assert state.is_done()
    assert state.wait_done(timeout=0.01)


def test_shared_producer_state_cancel():
    state = SharedProducerState(total_frames=1)
    assert not state.cancel_requested()
    state.request_cancel()
    assert state.cancel_requested()


# ── SharedBufferCacheAdapter tests ──────────────────────────────


def test_shared_buffer_cache_adapter_interface():
    buf = SharedFrameBuffer(total_frames=10, module_side=9)
    state = SharedProducerState(10)
    adapter = SharedBufferCacheAdapter(buf, state)
    try:
        assert adapter.total_frames == 10
        assert adapter.module_side == 9
        assert adapter.valid_count == 0
        assert not adapter.is_done()

        img = np.full((9, 9), 255, dtype=np.uint8)
        img[2, 3] = 0
        packed = pack_module_image(img)

        for i in range(5):
            buf.put_packed(i, packed)
            state.mark_produced()

        assert adapter.valid_count == 5
        assert adapter.has_frame(0)
        assert adapter.has_frame(4)
        assert not adapter.has_frame(5)
        assert adapter.contiguous_from(0) == 5
        assert adapter.contiguous_from(3) == 2

        module_img = adapter.get_module_image(0)
        assert module_img is not None
        assert np.array_equal(module_img, img)

        state.mark_done()
        assert adapter.is_done()
    finally:
        buf.close()
