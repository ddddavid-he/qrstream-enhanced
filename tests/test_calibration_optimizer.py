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


def test_optimizer_does_not_monotonically_suppress_fps_cadence_gain():
    version_stats = {40: stats_from_rate(1.0, 200)}
    fps_stats = {
        25: stats_from_rate(0.65, 200),
        30: stats_from_rate(0.83, 200),
    }

    result = optimize_calibration(
        version_stats,
        fps_stats,
        config=OptimizerConfig(fps_anchor_version=40),
    )

    balanced = result["balanced"]
    assert balanced is not None
    assert balanced.fps == 30
    assert balanced.overhead < 2.0


def test_optimizer_uses_pairwise_interpolation_when_exact_pair_missing():
    """When the picked (V,F) is bracketed by measured pairs, interpolate."""
    version_stats = {
        25: stats_from_rate(0.99, 200),
        40: stats_from_rate(0.99, 200),
    }
    fps_stats = {
        30: stats_from_rate(0.99, 200),
    }
    # Only off-axis corner pairs measured. The optimizer's only feasible
    # FPS is 30 and only versions are 25 and 40, so neither (25,30) nor
    # (40,30) is in pair_stats → must interpolate.
    pair_stats = {
        (25, 10): stats_from_rate(0.97, 200),
        (40, 10): stats_from_rate(0.95, 200),
        (25, 60): stats_from_rate(0.95, 200),
        (40, 60): stats_from_rate(0.93, 200),
    }

    result = optimize_calibration(
        version_stats,
        fps_stats,
        pair_stats=pair_stats,
        config=OptimizerConfig(fps_anchor_version=25),
    )

    aggressive = result["aggressive"]
    assert aggressive is not None
    assert aggressive.source == "pairwise-interp"


def test_optimizer_clamps_interpolation_at_grid_edges():
    """Queries outside the measurement grid clamp; no extrapolation."""
    pair_stats = {
        (25, 10): stats_from_rate(0.99, 200),
        (40, 10): stats_from_rate(0.99, 200),
    }
    # Only one fps measured (10), two versions. Query at fps=60 must clamp
    # to fps=10 (its only measurement axis) and interpolate over V.
    version_stats = {
        25: stats_from_rate(0.99, 200),
        40: stats_from_rate(0.99, 200),
    }
    fps_stats = {
        10: stats_from_rate(0.99, 200),
        60: stats_from_rate(0.99, 200),
    }

    result = optimize_calibration(
        version_stats,
        fps_stats,
        pair_stats=pair_stats,
        config=OptimizerConfig(fps_anchor_version=25),
    )

    # Some tier must surface interpolation as the source.
    sources = {tier: cand.source for tier, cand in result.items()
               if cand is not None}
    assert "pairwise-interp" in sources.values()


def test_optimizer_falls_back_to_separable_with_insufficient_pairs():
    """A single (V, F) measurement is not enough to interpolate."""
    pair_stats = {
        (25, 10): stats_from_rate(0.99, 200),
    }
    version_stats = {
        25: stats_from_rate(0.99, 200),
        40: stats_from_rate(0.99, 200),
    }
    fps_stats = {
        10: stats_from_rate(0.99, 200),
        30: stats_from_rate(0.99, 200),
    }

    result = optimize_calibration(
        version_stats,
        fps_stats,
        pair_stats=pair_stats,
        config=OptimizerConfig(fps_anchor_version=25),
    )

    # Pick will be (V40, 30fps); no interpolation possible (only 1 V and 1
    # F measured in pair_stats) → must fall back to separable.
    aggressive = result["aggressive"]
    assert aggressive is not None
    assert aggressive.source in {"separable", "fallback"}


def test_optimizer_respects_success_target_override():
    """Bumping the success target forces overhead to climb."""
    # Pick a moderate p so that aggressive (target 0.90) and the
    # 0.999 override land on different ladder rungs.
    version_stats = {40: stats_from_rate(0.70, 200)}
    fps_stats = {30: stats_from_rate(0.70, 200)}

    baseline = optimize_calibration(
        version_stats,
        fps_stats,
        config=OptimizerConfig(fps_anchor_version=40),
    )
    strict = optimize_calibration(
        version_stats,
        fps_stats,
        config=OptimizerConfig(
            fps_anchor_version=40, success_target_override=0.999),
    )

    base_aggr = baseline["aggressive"]
    strict_aggr = strict["aggressive"]
    assert base_aggr is not None
    assert strict_aggr is not None
    # A strictly higher success target cannot reduce required overhead.
    assert strict_aggr.overhead >= base_aggr.overhead
    # And on this rate the strict run must in fact bump it.
    assert strict_aggr.overhead > base_aggr.overhead
