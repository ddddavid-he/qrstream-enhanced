"""
Unit tests for probe-time adaptation helpers.

Pure/synthetic: no video I/O and no real QR detector.
"""

import threading

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


def test_probe_results_include_phase1_unique_blocks(monkeypatch):
    phase1 = [
        ProbeObservation(10, b"p1", 101, 100, True, 200.0, 1000, 1000, 500.0, 500.0),
    ]
    phase3 = [
        ProbeObservation(100, b"p3", 201, 100, True, 200.0, 1000, 1000, 500.0, 500.0),
        ProbeObservation(101, None, None, 100, True, 0.0, 1000, 1000, 0.0, 0.0),
    ]

    monkeypatch.setattr(dec, "_get_video_info", lambda _path: (500, 30.0, 1000, 1000))
    monkeypatch.setattr(dec, "_build_phase_burst_ranges", lambda *a, **k: [(0, 0)])
    monkeypatch.setattr(dec, "_build_probe_ranges", lambda *a, **k: [(100, 101)])

    def _fake_read_frame_ranges(_video_path, ranges, **_kwargs):
        if ranges == [(0, 0)]:
            yield (10, object())
        else:
            yield (100, object())
            yield (101, object())

    monkeypatch.setattr(dec, "_read_frame_ranges", _fake_read_frame_ranges)
    monkeypatch.setattr(
        dec,
        "_worker_probe_detect",
        lambda fd: phase1[0] if fd[0] == 10 else phase3[fd[0] - 100],
    )
    monkeypatch.setattr(dec, "_derive_crop_box", lambda *a, **k: None)
    monkeypatch.setattr(dec, "_extract_probe_video_constants", lambda *_a, **_k: None)
    monkeypatch.setattr(dec, "_adaptive_max_dim_from_probe", lambda *_a, **_k: None)
    monkeypatch.setattr(dec, "_analyze_probe_window", lambda *a, **k: {
        'frame_count': 2,
        'detect_rate': 0.5,
        'avg_repeat': 1.0,
        'distinct_seed_count': 1,
        'sample_rate': 1,
    })

    real_thread = threading.Thread

    class _InlineThread:
        def __init__(self, target=None, daemon=None):
            self._target = target
            self.daemon = daemon

        def start(self):
            if self._target is not None:
                self._target()

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(dec, "Thread", _InlineThread)

    try:
        sample_rate, probe_results, probe_count, leading, *_rest = dec._probe_sample_rate(
            "dummy.mp4", workers=1, verbose=False,
        )
    finally:
        monkeypatch.setattr(dec, "Thread", real_thread)

    assert sample_rate == 1
    assert probe_count == 2
    assert leading == 0
    assert probe_results == [
        (10, b"p1", 101),
        (100, b"p3", 201),
        (101, None, None),
    ]


def test_scan_progress_tracker_accounts_for_skips_and_hits():
    class _Reporter:
        def __init__(self):
            self.calls = []

        def scan_update(self, **kwargs):
            self.calls.append(kwargs)

    reporter = _Reporter()
    decoder = dec.LTDecoder()
    tracker = dec._ScanProgressTracker(
        total_frames=100,
        leading_frames_probed=10,
        lt_decoder=decoder,
        reporter=reporter,
    )

    tracker.mark_skipped_until(14)
    tracker.on_frame(14, True)

    assert reporter.calls
    update = reporter.calls[-1]
    assert update["video_pct"] == 15.0
    assert update["hit_window"] == 1.0
    assert update["file_pct"] == 0.0
    assert update["k"] is None


def test_tracked_read_frames_marks_skipped_positions(monkeypatch):
    seen = []

    def _fake_read_frames(*_args, **_kwargs):
        yield (5, "a")
        yield (8, "b")

    class _Tracker:
        def mark_skipped_until(self, frame_idx):
            seen.append(frame_idx)

    monkeypatch.setattr(dec, "_read_frames", _fake_read_frames)

    frames = list(dec._tracked_read_frames(
        "dummy.mp4",
        2,
        100,
        start_frame=0,
        max_detect_dim=720,
        crop_box=None,
        scan_tracker=_Tracker(),
    ))

    assert frames == [(5, "a"), (8, "b")]
    assert seen == [5, 8]
