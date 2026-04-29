"""
Unit tests for the per-video PPM-threshold learning algorithm.

Tests :func:`qrstream.decoder._learn_ppm_threshold` with synthetic
hit-rate curves and edge cases.  These are pure-function tests — no
video I/O, no QR detection — so they run in milliseconds.
"""

import random

import pytest

from qrstream.decoder import (
    _learn_ppm_threshold,
    _PPM_LEARN_MIN_SAMPLES,
    _PPM_LEARN_SAFETY_MARGIN,
)


# ── helpers ──────────────────────────────────────────────────────

def _make_samples(
    n: int,
    step_ppm: float,
    below_rate: float,
    above_rate: float,
    ppm_range: tuple[float, float] = (1.0, 12.0),
    seed: int = 42,
) -> list[tuple[float, bool]]:
    """Generate ``n`` synthetic ``(ppm, decoded)`` samples.

    Below ``step_ppm`` the decode probability is ``below_rate``;
    above it the probability is ``above_rate``.  PPM values are
    uniformly distributed over ``ppm_range``.
    """
    rng = random.Random(seed)
    lo, hi = ppm_range
    samples = []
    for _ in range(n):
        ppm = rng.uniform(lo, hi)
        p = below_rate if ppm < step_ppm else above_rate
        samples.append((ppm, rng.random() < p))
    return samples


# ── inflection detection tests ───────────────────────────────────

class TestLearnPpmThreshold:
    """Core algorithm correctness."""

    def test_sharp_step_at_3(self):
        """Sharp cliff at ppm=3: below 5%, above 90%.

        Expected learned ≈ 3.0 × safety_margin.
        """
        samples = _make_samples(200, step_ppm=3.0,
                                below_rate=0.05, above_rate=0.90,
                                ppm_range=(1.0, 10.0))
        result = _learn_ppm_threshold(samples)
        assert result is not None
        assert 2.5 <= result <= 5.0, f"expected ~3.45, got {result}"

    def test_step_at_5(self):
        """Step at ppm=5: below 30%, above 80%."""
        samples = _make_samples(200, step_ppm=5.0,
                                below_rate=0.30, above_rate=0.80,
                                ppm_range=(2.0, 8.0))
        result = _learn_ppm_threshold(samples)
        assert result is not None
        assert 4.5 <= result <= 7.0, f"expected ~5.75, got {result}"

    def test_gradual_ramp(self):
        """Gradual 4-tier ramp: 0% → 40% → 70% → 80%.

        The transition spans ppm 3–5; learned should land in that
        range (with safety margin).
        """
        rng = random.Random(7)
        samples = []
        for _ in range(200):
            ppm = rng.uniform(2.0, 10.0)
            if ppm < 3:
                p = 0.0
            elif ppm < 4:
                p = 0.40
            elif ppm < 5:
                p = 0.70
            else:
                p = 0.80
            samples.append((ppm, rng.random() < p))

        result = _learn_ppm_threshold(samples)
        assert result is not None
        assert 3.0 <= result <= 7.0, f"expected ~4-5 range, got {result}"

    def test_all_pass_returns_low_threshold(self):
        """When every sample decodes, the learned ppm should be very low."""
        samples = [(p / 10.0, True) for p in range(50)]
        result = _learn_ppm_threshold(samples)
        assert result is not None
        assert result < 2.0, f"expected low, got {result}"


# ── edge case tests ──────────────────────────────────────────────

class TestLearnPpmEdgeCases:

    def test_returns_none_when_too_few_samples(self):
        samples = [(2.0, True)] * (_PPM_LEARN_MIN_SAMPLES - 1)
        assert _learn_ppm_threshold(samples) is None

    def test_returns_none_when_all_fail(self):
        """All detections fail → plateau < 0.3 → None."""
        samples = [(float(i), False) for i in range(50)]
        assert _learn_ppm_threshold(samples) is None

    def test_returns_none_when_plateau_below_threshold(self):
        """Even the best windows have < 30% hit rate."""
        rng = random.Random(99)
        samples = [(rng.uniform(1, 10), rng.random() < 0.10) for _ in range(100)]
        assert _learn_ppm_threshold(samples) is None

    def test_exactly_min_samples(self):
        """Boundary: exactly _PPM_LEARN_MIN_SAMPLES samples."""
        n = _PPM_LEARN_MIN_SAMPLES
        samples = _make_samples(n, step_ppm=4.0,
                                below_rate=0.0, above_rate=1.0,
                                ppm_range=(2.0, 8.0))
        result = _learn_ppm_threshold(samples)
        # With only 30 samples of a clean step, should still find it.
        assert result is not None
        assert 3.0 <= result <= 6.0

    def test_safety_margin_applied(self):
        """Verify the safety margin is actually reflected in the output.

        With a perfectly clean step at ppm=4.0 (0% below, 100% above),
        the raw inflection should be ~4.0.  The learned value should be
        noticeably higher due to the safety margin.
        """
        samples = _make_samples(200, step_ppm=4.0,
                                below_rate=0.0, above_rate=1.0,
                                ppm_range=(2.0, 8.0))
        result = _learn_ppm_threshold(samples)
        assert result is not None
        # Must be strictly above 4.0 (the raw inflection)
        assert result > 4.0
        # And below 4.0 × margin × generous tolerance
        assert result < 4.0 * _PPM_LEARN_SAFETY_MARGIN * 1.3
