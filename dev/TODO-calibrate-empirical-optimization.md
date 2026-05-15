# TODO: Calibrate as Bounded Discrete Empirical Optimization

This document records the current branch discussion about the future direction of
`qrstream calibrate`. It is intended as implementation context for future
contributors.

## Background

The current `calibrate` workflow samples a real capture video and uses observed
QR detection rates to recommend encode parameters. Conceptually, this is close to
an optimization problem over encode parameters, but the implementation is not a
strict linear optimizer.

The current recommendation model searches over tested QR versions and FPS values,
then derives an overhead value from observed detection quality and tier-specific
safety margins.

## Current Interpretation

A useful model is:

```text
Decision variables:
  V = QR version
  F = video FPS
  O = LT/fountain overhead

Estimated from calibration video:
  p_v(V) = detection rate for QR version V
  p_f(F) = detection rate for FPS F
```

Current tier constraints are cumulative:

```text
safe       : min(p_v(V), p_f(F)) >= 0.90
balanced   : min(p_v(V), p_f(F)) >= 0.80
aggressive : min(p_v(V), p_f(F)) >= 0.70
```

The current overhead recommendation is derived, not independently optimized:

```text
combined_dr = p_v(V) * p_f(F)
raw_overhead = 1 / combined_dr
overhead = max(raw_overhead * safety_margin[tier], MIN_OVERHEAD)
```

Current safety margins:

```text
safe       : 1.30
balanced   : 1.15
aggressive : 1.05
```

The current objective can be described as:

```text
maximize throughput(V, F, O)
subject to tier constraint for (V, F)
```

where `O` is derived from `(V, F, tier)`, so the actual current search is closer
to:

```text
for each tier:
  choose best (V, F) among valid candidates
  derive O from observed detection rates
```

## Why This Is Not Strict Linear Optimization

Although the problem is bounded and parameterized, it is not currently a linear
optimization problem:

1. The candidate space is discrete.
   - QR version is selected from a preset ladder.
   - FPS is selected from a preset ladder and may be capped by display or capture
     metadata.

2. The objective is nonlinear.
   - Throughput is roughly proportional to QR capacity and FPS, and inversely
     proportional to overhead:

   ```text
   throughput ~= capacity(V) * F / O
   ```

   QR capacity is not linear in version, and `O` is derived from inverse detection
   probability.

3. The constraints are nonlinear.
   - Tier eligibility uses `min(p_v, p_f)`.
   - Overhead uses `1 / (p_v * p_f)`.

4. `O` is not yet a first-class search variable.
   - Current implementation searches `(V, F)` and derives `O`.
   - A true 3-parameter optimizer would search or solve for `(V, F, O)` jointly.

5. The current model assumes separability.
   - Version quality and FPS quality are estimated separately, then combined.
   - Real captures may have interaction effects: high QR version at high FPS may
     suffer from motion blur, exposure, rolling shutter, compression artifacts, or
     decoder limits in ways that are not captured by independent `p_v` and `p_f`.

A more precise description of the current system is therefore:

```text
bounded discrete empirical optimization with derived overhead
```

or:

```text
two-dimensional discrete search over (QR version, FPS), with overhead derived
from empirical detection rates and tier safety margin
```

## Future Direction

A future implementation could promote `overhead` to a true optimization variable
and model calibration as a constrained 3-parameter optimization problem:

```text
maximize throughput(V, F, O)
subject to estimated_success_probability(V, F, O) >= target_success[tier]
```

Possible target success levels:

```text
safe       : highest reliability target
balanced   : medium reliability target
aggressive : lower reliability target
```

This would allow `calibrate` to reason directly about the tradeoff between:

- QR version: density/capacity vs detectability
- FPS: temporal throughput vs capture stability
- Overhead: redundancy/recovery probability vs effective throughput

## Possible Implementation Plan

### 1. Introduce an explicit calibration optimizer module

Create a focused abstraction, for example:

```python
@dataclass(frozen=True)
class CalibrationCandidate:
    qr_version: int
    fps: int
    overhead: float
    estimated_success: float
    estimated_throughput: float
```

and an optimizer entry point such as:

```python
def optimize_calibration(
    observations: CalibrationObservations,
    constraints: CalibrationConstraints,
    tier: CalibrationTier,
) -> CalibrationCandidate | None:
    ...
```

This would separate:

- measurement/parsing from captured video
- probability modeling
- optimization/search
- CLI rendering

### 2. Make overhead a bounded search dimension

Instead of deriving one overhead per `(V, F, tier)`, enumerate or solve over a
bounded overhead ladder, for example:

```text
O in [1.05, 1.10, 1.15, 1.20, 1.30, 1.50, 1.80, 2.00]
```

or use a continuous solver later if the probability model becomes smooth enough.

The optimizer can then choose the minimum overhead that satisfies the target
success constraint, rather than applying a fixed safety multiplier.

### 3. Improve the success probability model

Current approximation:

```text
combined_dr = p_v(V) * p_f(F)
```

Future alternatives:

```text
p_frame(V, F) = learned/interpolated frame detection probability
p_decode_success(V, F, O, file_size, block_size) = fountain recovery probability
```

A more realistic model should account for:

- file size / block count
- LT overhead and expected recovery threshold
- frame loss distribution, not only average frame detection rate
- burst losses from camera autofocus, exposure changes, or hand motion
- interaction between QR version and FPS
- capture video FPS ceiling and actual frame sampling behavior

### 4. Consider direct pairwise calibration samples

Current calibration estimates version and FPS separately. A future calibration
sequence could include selected pairwise probes:

```text
(V25, 10fps), (V30, 15fps), (V35, 30fps), (V40, 60fps), ...
```

This would measure `p(V, F)` directly for important candidate pairs and reduce
reliance on separability assumptions.

A practical compromise:

1. Keep current independent ladders for broad coverage.
2. Use them to identify a promising frontier.
3. Add a short second-stage pairwise probe near the frontier.

### 5. Use video metadata as hard and soft constraints

Existing metadata already available from captured video includes:

- width
- height
- FPS
- frame count
- duration

Future optimization should use these as constraints/signals:

```text
F <= captured_video_fps_ceiling
```

Potential additional heuristics:

- If resolution is low, penalize high QR versions.
- If capture FPS is low or variable, avoid recommending high FPS even if sparse
  samples decode occasionally.
- If frame count is too small for a preset, mark observations as low confidence.

### 6. Represent confidence separately from quality

The current result exposes channel quality and tier recommendations. Future work
should distinguish:

```text
quality    = observed detectability
confidence = statistical confidence in the estimate
```

For example, a 100% detection rate over 5 samples should not be treated the same
as a 95% detection rate over 200 samples.

Potential techniques:

- Wilson score interval
- beta-binomial posterior
- minimum sample thresholds per candidate
- confidence-adjusted detection rate for tier eligibility

### 7. Preserve current UX semantics

Future changes should keep these user-facing expectations unless deliberately
changed:

- `calibrate` with no mode defaults to display mode.
- Public presets start from encode defaults and explore upward.
- Weak channels should be calibrated with `--precision low`.
- `safe`, `balanced`, and `aggressive` are cumulative reliability tiers.
- High-quality channels should normally produce all three tiers, with different
  overhead/safety tradeoffs.

## Open Questions

1. Should `Channel quality` incorporate FPS stability, or remain version-only?
2. Should `overhead` optimize for expected decode success at a specific file size?
3. Should calibration ask for or infer target file size?
4. Should the optimizer prefer lower overhead or higher throughput when two
   candidates have similar estimated success?
5. How much extra calibration duration is acceptable for pairwise second-stage
   probing?

## Suggested Next Step

Implement a pure, testable optimizer function that accepts synthetic observation
maps and returns tier recommendations. Keep the existing CLI behavior unchanged,
then migrate the current recommendation logic into that optimizer before adding a
true overhead search dimension.
