import pytest

from qrstream.calibration_optimizer import (
    DEFAULT_TARGET_K,
    DetectionStats,
    OptimizerConfig,
    estimate_success_probability,
    monotonic_envelope,
    optimize_calibration,
    stats_from_rate,
    wilson_lower_bound,
)


def test_default_target_k_is_long_file_scale():
    assert DEFAULT_TARGET_K == 1000
    assert OptimizerConfig().target_k == 1000


def test_wilson_lower_bound_penalizes_small_samples():
    assert wilson_lower_bound(5, 5, z=1.645) < 0.75
    assert wilson_lower_bound(95, 100, z=1.645) > 0.88


@pytest.mark.parametrize("z", [0.0, 1.0, 1.645])
def test_wilson_lower_bound_is_clamped(z):
    assert wilson_lower_bound(0, 10, z=z) == 0.0
    assert 0.0 <= wilson_lower_bound(10, 10, z=z) <= 1.0


def test_monotonic_envelope_is_conservative():
    stats = {
        25: stats_from_rate(0.90, 100),
        30: stats_from_rate(0.80, 100),
        35: stats_from_rate(0.85, 100),
    }

    smoothed = monotonic_envelope(stats)

    assert smoothed[25].raw_rate == pytest.approx(0.90)
    assert smoothed[30].raw_rate == pytest.approx(0.80)
    assert smoothed[35].raw_rate == pytest.approx(0.80)


def test_success_probability_improves_with_overhead():
    low = estimate_success_probability(1000, 1.10, 0.90)
    high = estimate_success_probability(1000, 1.30, 0.90)

    assert low < high
    assert high > 0.99


def test_success_probability_requires_enough_frames():
    assert estimate_success_probability(1000, 1.0, 1.0) == 0.0
    assert estimate_success_probability(1000, 1.05, 1.0) == 1.0


def test_optimizer_picks_highest_throughput_feasible_candidate():
    version_stats = {
        25: stats_from_rate(0.99, 200),
        35: stats_from_rate(0.97, 200),
    }
    fps_stats = {
        10: stats_from_rate(0.99, 200),
        30: stats_from_rate(0.96, 200),
    }

    result = optimize_calibration(
        version_stats,
        fps_stats,
        config=OptimizerConfig(fps_anchor_version=25),
    )

    safe = result["safe"]
    assert safe is not None
    assert safe.qr_version == 35
    assert safe.fps == 30
    assert safe.estimated_success >= 0.99
    assert safe.overhead >= 1.05


def test_optimizer_prefers_pairwise_observation():
    version_stats = {
        25: stats_from_rate(0.99, 200),
        40: stats_from_rate(0.99, 200),
    }
    fps_stats = {
        30: stats_from_rate(0.99, 200),
    }
    pair_stats = {
        (40, 30): stats_from_rate(0.50, 200),
        (25, 30): stats_from_rate(0.99, 200),
    }

    result = optimize_calibration(
        version_stats,
        fps_stats,
        pair_stats=pair_stats,
        config=OptimizerConfig(fps_anchor_version=25),
    )

    safe = result["safe"]
    assert safe is not None
    assert safe.qr_version == 25
    assert safe.source == "pairwise"


def test_optimizer_honors_capture_fps_ceiling():
    version_stats = {40: DetectionStats(200, 200)}
    fps_stats = {
        30: DetectionStats(200, 200),
        60: DetectionStats(200, 200),
    }

    result = optimize_calibration(
        version_stats,
        fps_stats,
        config=OptimizerConfig(capture_fps_ceiling=30),
    )

    safe = result["safe"]
    assert safe is not None
    assert safe.fps == 30
