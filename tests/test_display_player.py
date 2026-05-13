"""Tests for display-only player helpers."""

import numpy as np

from qrstream.display_cache import ModuleFrameCache
from qrstream.display_player import (
    DisplayProducerState,
    _STATUS_PANEL_HEIGHT,
    _compose_status_frame,
    _seek_if_cached,
)


def test_status_panel_does_not_modify_qr_region():
    cache = ModuleFrameCache(total_frames=1, module_side=9)
    cache.put_module_image(0, np.full((9, 9), 255, dtype=np.uint8))
    cache.mark_done()
    state = DisplayProducerState(total_frames=1)
    state.mark_done()

    qr_frame = np.zeros((90, 90, 3), dtype=np.uint8)
    canvas = _compose_status_frame(
        qr_frame,
        frame_index=0,
        cache=cache,
        state=state,
        fps=10,
        playing=False,
        can_play=True,
    )

    assert canvas.shape == (90 + _STATUS_PANEL_HEIGHT, 90, 3)
    assert np.array_equal(canvas[:90, :90], qr_frame)
    assert np.any(canvas[90:, :] != 255)


def test_seek_only_moves_to_cached_frames():
    cache = ModuleFrameCache(total_frames=4, module_side=9)
    img = np.full((9, 9), 255, dtype=np.uint8)
    cache.put_module_image(0, img)
    cache.put_module_image(2, img)

    assert _seek_if_cached(cache, 0, 2) == 2
    assert _seek_if_cached(cache, 2, 3) == 2
    assert _seek_if_cached(cache, 2, -10) == 0
