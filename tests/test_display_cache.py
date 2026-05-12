"""Tests for display-only frame caches."""

import numpy as np

from qrstream.display_cache import (
    ModuleFrameCache,
    PresentationFrameCache,
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


def test_presentation_cache_enforces_budget():
    cache = PresentationFrameCache(budget_bytes=100)
    frame = np.zeros((5, 5, 3), dtype=np.uint8)  # 75 bytes

    cache.put((0, 5), frame)
    cache.put((1, 5), frame)

    assert cache.current_bytes <= 100
    assert cache.get((0, 5)) is None
    assert np.array_equal(cache.get((1, 5)), frame)
