"""Three-parameter calibration optimizer for QRStream.

This module is intentionally pure: it does not decode videos or format UI.
It turns calibration probe statistics into candidate encode parameters by
searching QR version, FPS, and fountain-code overhead jointly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .protocol import _alphanumeric_byte_capacity

DEFAULT_TARGET_K = 1000
RQ_DECODE_MARGIN = 0.015

RQ_OVERHEAD_LADDER = (
    1.05, 1.10, 1.15, 1.20, 1.25, 1.30,
    1.35, 1.40, 1.50, 1.60, 1.80, 2.00,
)
LT_OVERHEAD_LADDER = (
    1.20, 1.30, 1.40, 1.50, 1.70, 2.00, 2.50, 3.00,
)

TIER_SUCCESS_TARGETS = {
    "safe": 0.99,
    "balanced": 0.95,
    "aggressive": 0.90,
}

TIER_WILSON_Z = {
    "safe": 1.645,
    "balanced": 1.0,
    "aggressive": 0.0,
}


@dataclass(frozen=True)
class DetectionStats:
    """Detection counts and burst metadata for one calibration probe."""

    detected: int
    total: int
    max_miss_run: int = 0
    mean_miss_run: float = 0.0

    @property
    def raw_rate(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(max(self.detected / self.total, 0.0), 1.0)

    @property
    def wilson_lower(self) -> float:
        return wilson_lower_bound(self.detected, self.total, z=1.0)

    def lower_bound(self, z: float) -> float:
        return wilson_lower_bound(self.detected, self.total, z=z)


@dataclass(frozen=True)
class CalibrationCandidate:
    """One feasible three-parameter recommendation."""

    qr_version: int
    fps: int
    overhead: float
    frame_detect_probability: float
    estimated_success: float
    estimated_throughput_bps: float
    source: str


@dataclass(frozen=True)
class OptimizerConfig:
    """Inputs that shape calibration optimization."""

    codec: str = "raptorq"
    target_k: int = DEFAULT_TARGET_K
    capture_fps_ceiling: int | None = None
    fps_anchor_version: int | None = None
    rq_decode_margin: float = RQ_DECODE_MARGIN
    overhead_ladder: tuple[float, ...] | None = None
    ec_level: int = 1

    @property
    def resolved_overhead_ladder(self) -> tuple[float, ...]:
        if self.overhead_ladder is not None:
            return self.overhead_ladder
        if self.codec == "lt":
            return LT_OVERHEAD_LADDER
        return RQ_OVERHEAD_LADDER


def make_detection_stats(detected: int, total: int) -> DetectionStats:
    """Build DetectionStats from raw counts, clamping invalid inputs."""
    total = max(0, int(total))
    detected = min(max(0, int(detected)), total)
    return DetectionStats(detected=detected, total=total)


def stats_from_rate(rate: float, total: int) -> DetectionStats:
    """Approximate DetectionStats when only a historical float rate exists."""
    total = max(1, int(total))
    clamped = min(max(float(rate), 0.0), 1.0)
    return make_detection_stats(round(clamped * total), total)


def wilson_lower_bound(detected: int, total: int, z: float = 1.0) -> float:
    """Wilson score lower bound for a binomial proportion."""
    if total <= 0:
        return 0.0
    detected = min(max(0, detected), total)
    if z <= 0:
        return detected / total

    phat = detected / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = phat + z2 / (2.0 * total)
    spread = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * total)) / total)
    return max(0.0, min(1.0, (centre - spread) / denom))


def monotonic_envelope(stats: Mapping[int, DetectionStats]) -> dict[int, DetectionStats]:
    """Conservative non-increasing envelope over ascending parameter values."""
    smoothed: dict[int, DetectionStats] = {}
    previous_rate: float | None = None
    for key in sorted(stats):
        current = stats[key]
        raw_rate = current.raw_rate
        rate = raw_rate if previous_rate is None else min(raw_rate, previous_rate)
        previous_rate = rate
        smoothed[key] = stats_from_rate(rate, current.total)
    return smoothed


def estimate_success_probability(
    target_k: int,
    overhead: float,
    frame_detect_probability: float,
    rq_decode_margin: float = RQ_DECODE_MARGIN,
) -> float:
    """Estimate fountain decode success over an erasure channel."""
    k = max(1, int(target_k))
    n = max(0, math.ceil(k * overhead))
    required = math.ceil(k * (1.0 + rq_decode_margin))
    p = min(max(frame_detect_probability, 0.0), 1.0)

    if required <= 0:
        return 1.0
    if required > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    return _binomial_sf(required, n, p)


def optimize_calibration(
    version_stats: Mapping[int, DetectionStats],
    fps_stats: Mapping[int, DetectionStats],
    pair_stats: Mapping[tuple[int, int], DetectionStats] | None = None,
    config: OptimizerConfig | None = None,
) -> dict[str, CalibrationCandidate | None]:
    """Search QR version, FPS, and overhead for each risk tier."""
    if config is None:
        config = OptimizerConfig()
    pair_stats = pair_stats or {}

    versions = sorted(version_stats)
    fps_values = sorted(fps_stats)
    if config.capture_fps_ceiling is not None:
        fps_values = [fps for fps in fps_values if fps <= config.capture_fps_ceiling]

    version_env = monotonic_envelope(version_stats)
    fps_observed = dict(fps_stats)

    results: dict[str, CalibrationCandidate | None] = {}
    for tier, target_success in TIER_SUCCESS_TARGETS.items():
        z = TIER_WILSON_Z[tier]
        best: tuple[float, float, int, int, float, CalibrationCandidate] | None = None

        for version in versions:
            for fps in fps_values:
                p_frame, source = _estimate_frame_probability(
                    version, fps, version_env, fps_observed, pair_stats, config, z)
                if p_frame <= 0.0:
                    continue

                for overhead in config.resolved_overhead_ladder:
                    success = estimate_success_probability(
                        config.target_k, overhead, p_frame, config.rq_decode_margin)
                    if success < target_success:
                        continue

                    throughput = _estimate_throughput(
                        version, fps, overhead, config.ec_level)
                    candidate = CalibrationCandidate(
                        qr_version=version,
                        fps=fps,
                        overhead=overhead,
                        frame_detect_probability=p_frame,
                        estimated_success=success,
                        estimated_throughput_bps=throughput,
                        source=source,
                    )
                    ranking = (throughput, success, version, fps, -overhead, candidate)
                    if best is None or ranking > best:
                        best = ranking
                    break

        results[tier] = best[-1] if best is not None else None

    return results


def _estimate_frame_probability(
    version: int,
    fps: int,
    version_stats: Mapping[int, DetectionStats],
    fps_stats: Mapping[int, DetectionStats],
    pair_stats: Mapping[tuple[int, int], DetectionStats],
    config: OptimizerConfig,
    z: float,
) -> tuple[float, str]:
    pair = pair_stats.get((version, fps))
    if pair is not None:
        return pair.lower_bound(z), "pairwise"

    version_stat = version_stats.get(version)
    fps_stat = fps_stats.get(fps)
    if version_stat is None or fps_stat is None:
        return 0.0, "missing"

    p_v = version_stat.lower_bound(z)
    p_f = fps_stat.lower_bound(z)
    if p_v <= 0.0 or p_f <= 0.0:
        return 0.0, "separable"

    anchor = None
    if config.fps_anchor_version is not None:
        anchor = version_stats.get(config.fps_anchor_version)
    p_anchor = anchor.lower_bound(z) if anchor is not None else 0.0

    if p_anchor < 0.5:
        p = min(p_v, p_f)
        return max(0.0, min(p, 1.0)), "fallback"

    p = p_v * p_f / p_anchor
    p = max(0.0, min(p, p_v, p_f, 1.0))
    return p, "separable"


def _estimate_throughput(qr_version: int, fps: int, overhead: float,
                         ec_level: int) -> float:
    if not 1 <= qr_version <= 40 or fps <= 0 or overhead <= 0:
        return 0.0
    capacity = _alphanumeric_byte_capacity(qr_version, ec_level)
    return capacity * fps / overhead


def _binomial_sf(k: int, n: int, p: float) -> float:
    """Survival function P[X >= k] for X~Binomial(n,p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0

    mean = n * p
    if k > mean:
        return _sum_binomial_range(k, n, n, p)

    cdf = _sum_binomial_range(0, k - 1, n, p)
    return max(0.0, min(1.0, 1.0 - cdf))


def _sum_binomial_range(start: int, stop: int, n: int, p: float) -> float:
    if start > stop:
        return 0.0
    log_terms = [_binomial_logpmf(i, n, p) for i in range(start, stop + 1)]
    max_log = max(log_terms)
    if max_log == -math.inf:
        return 0.0
    total = sum(math.exp(term - max_log) for term in log_terms)
    return max(0.0, min(1.0, math.exp(max_log) * total))


def _binomial_logpmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
