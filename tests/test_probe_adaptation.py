"""
Unit tests for the probe-time adaptive crop ROI (continuous margin).

Pure/synthetic: no video I/O and no real QR detector.
"""

from qrstream import decoder as dec
from qrstream.decoder import ProbeObservation


def _obs(frame_idx: int, cx: float, cy: float, side: float = 200.0) -> ProbeObservation:
    return ProbeObservation(
        frame_idx,
        b"payload",
        frame_idx,
        100,
        True,
        side,
        1000,
        1000,
        cx,
        cy,
    )


def _expected_crop_size(med_side: float, norm_iqr: float) -> int:
    return int(med_side * dec._crop_margin_for_jitter(norm_iqr) / 2) * 2


def test_crop_margin_for_jitter_is_monotonic():
    thresh = dec._CROP_STABILITY_THRESHOLD
    values = [0.0, thresh * 0.25, thresh * 0.5, thresh * 0.75, thresh]
    margins = [dec._crop_margin_for_jitter(v) for v in values]
    assert margins[0] == dec._CROP_MARGIN_MIN
    assert margins[-1] == dec._CROP_MARGIN_MAX
    for a, b in zip(margins, margins[1:]):
        assert a <= b


def test_crop_margin_clamps_above_threshold():
    assert dec._crop_margin_for_jitter(1.0) == dec._CROP_MARGIN_MAX


def test_derive_crop_box_uses_near_min_margin_for_stable_qr():
    observations = [_obs(i, 500 + i, 500) for i in range(10)]

    crop = dec._derive_crop_box(observations, 1000, 1000)

    assert crop is not None
    y0, y1, x0, x1 = crop
    # q1=502, q3=507 → norm_iqr=0.005
    expected = _expected_crop_size(200, 0.005)
    assert (x1 - x0) == expected
    assert (y1 - y0) == expected


def test_derive_crop_box_margin_scales_continuously_with_position_jitter():
    observations = [_obs(i, 400 + i * 20, 500) for i in range(10)]

    crop = dec._derive_crop_box(observations, 1000, 1000)

    assert crop is not None
    y0, y1, x0, x1 = crop
    # q1=440, q3=540 → norm_iqr=0.10
    expected = _expected_crop_size(200, 0.10)
    assert (x1 - x0) == expected
    assert (y1 - y0) == expected


def test_derive_crop_box_margin_scales_continuously_with_side_jitter():
    observations = [_obs(i, 500, 500, side=180 + i * 5) for i in range(10)]

    crop = dec._derive_crop_box(observations, 1000, 1000)

    assert crop is not None
    y0, y1, x0, x1 = crop
    med_side = 180 + 5 * 5
    # q1=190, q3=215 → norm_iqr=25/205
    expected = _expected_crop_size(med_side, 25 / med_side)
    assert (x1 - x0) == expected
    assert (y1 - y0) == expected


def test_derive_crop_box_disables_crop_for_unstable_qr():
    observations = [_obs(i, 300 + i * 40, 500) for i in range(10)]

    assert dec._derive_crop_box(observations, 1000, 1000) is None
