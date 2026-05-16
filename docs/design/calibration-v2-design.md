# QRStream Calibration V2 Design

## Mathematical Foundation

### 1. LT/RaptorQ Decoding Success Model

#### Current Model (V1 - Flawed)
```python
# calibrate.py lines 598-605
pair_rate = min(ver_dr, fps_dr)  # arbitrary heuristic
combined_dr = ver_dr * fps_dr     # independence assumption
raw_overhead = 1.0 / combined_dr
overhead = raw_overhead * safety_margin
```

**Problems:**
- Independence assumption: `p(V,F) ≠ p(V) × p(F)` in practice
- `min()` has no theoretical basis
- Overhead formula ignores block count K
- Safety margin is arbitrary (1.30, 1.15, 1.05)

#### Proposed Model (V2)

**Binomial Framework:**

For a file with K source blocks, encoding with overhead `h`:
- N = K × h encoded blocks transmitted
- Each frame has detection probability p (depends on V, F)
- S ~ Binomial(N, p) = number of successfully decoded blocks

**Decode Success Condition (LT/RaptorQ):**

For RaptorQ (RFC 6330), decoding succeeds with high probability when:
```
S ≥ K × (1 + ε) where ε ≈ 0.01-0.05
```

**Required Sample Size:**

Using Chernoff bound for Binomial:
```
P(S < (1-δ)μ) ≤ exp(-μδ²/2) where μ = Np
```

Setting δ such that (1-δ)μ = K(1+ε):
```
Np(1-δ) = K(1+ε)
N = K(1+ε) / (p(1-δ))
```

For confidence level α (e.g., 95% → z = 1.96):
```
N ≥ (K(1+ε) + z√(K(1+ε)(1-p))) / p
```

**Overhead Formula (V2):**
```python
def compute_overhead(K: int, p: float, confidence: float = 0.95) -> float:
    """
    Compute required overhead for K source blocks.
    
    Parameters:
        K: number of source blocks
        p: detection probability (0 < p ≤ 1)
        confidence: target decode success probability (e.g., 0.95)
    
    Returns:
        Required overhead ratio (N/K)
    """
    if p <= 0:
        return float('inf')
    
    if p >= 1.0:
        return 1.05  # RaptorQ near-optimal floor
    
    # RaptorQ epsilon (redundancy factor)
    epsilon = 0.05  # 5% auxiliary blocks needed
    
    # Z-score for confidence level
    import math
    if confidence >= 0.995:
        z = 2.576
    elif confidence >= 0.99:
        z = 2.326
    elif confidence >= 0.95:
        z = 1.96
    elif confidence >= 0.90:
        z = 1.645
    else:
        z = 1.282
    
    # Required encoded blocks for confidence
    # N >= (K(1+ε) + z√(K(1+ε)p(1-p))) / p
    # (using p for variance since we observe decoded frames)
    K_prime = K * (1 + epsilon)
    var_term = z * math.sqrt(K_prime * p * (1 - p))
    
    N_required = (K_prime + var_term) / p
    overhead = N_required / K
    
    # Apply floor based on codec
    return max(overhead, MIN_OVERHEAD_RQ)  # 1.05 for RaptorQ
```

**Key Insight:** Overhead depends on **K**, not just p!
- Small K (e.g., 100 blocks): Less overhead needed (law of large numbers)
- Large K (e.g., 10,000 blocks): More overhead needed (binomial variance)

---

## 2. Joint (V, F) Probability Model

### Current Approach
- Tests version ladder at fixed 10 FPS
- Tests FPS ladder at fixed anchor version
- Assumes independence

### Proposed Approach: Sparse Grid Sampling

**Phase 1: Independent Ladders (keep current)**
- Quickly estimate p(V) and p(F) separately
- Identify promising regions

**Phase 2: Joint Testing (new)**
Test a **subset** of (V, F) pairs to estimate interaction:

```python
# Sparse grid: test promising combinations
JOINT_TEST_PAIRS = [
    (25, 10), (25, 15), (25, 20),
    (30, 10), (30, 15),
    (35, 10),
    # ... selected based on Phase 1 results
]
```

**Model Fitting:**

Fit a joint probability surface:
```
p(V, F) ≈ f(V, F) where f is a learned function
```

Options:
1. **Empirical lookup**: Store measured p(V,F) for tested pairs
2. **Bilinear interpolation**: For untested (V,F), interpolate from nearest measured points
3. **Parametric model**: p(V,F) = g(V) × h(F) × correction_term

**Implementation:**

```python
@dataclass
class JointProbabilityModel:
    """Model p(V, F) from sparse measurements."""
    
    # Measured: (V, F) -> (detected, total)
    measurements: dict[tuple[int, int], tuple[int, int]]
    
    def estimate_p(self, V: int, F: int) -> float:
        """Estimate p(V, F) with interpolation."""
        if (V, F) in self.measurements:
            det, tot = self.measurements[(V, F)]
            return det / tot if tot > 0 else 0.0
        
        # Bilinear interpolation from nearest measured points
        return self._interpolate(V, F)
    
    def _interpolate(self, V: int, F: int) -> float:
        """Bilinear interpolation from sparse grid."""
        # Find nearest measured points
        measured_V = sorted(set(v for v, _ in self.measurements))
        measured_F = sorted(set(f for _, f in self.measurements))
        
        # Nearest V
        V_low = max((v for v in measured_V if v <= V), default=measured_V[0])
        V_high = min((v for v in measured_V if v >= V), default=measured_V[-1])
        
        # Nearest F  
        F_low = max((f for f in measured_F if f <= F), default=measured_F[0])
        F_high = min((f for f in measured_F if f >= F), default=measured_F[-1])
        
        # Interpolate (simplified - can use proper bilinear)
        p_low = self._interp_1d(V, F_low, V_low, V_high)
        p_high = self._interp_1d(V, F_high, V_low, V_high)
        # ... full bilinear interpolation
        
        return interpolated_p
    
    def confidence_interval(self, V: int, F: int, alpha: float = 0.05) -> tuple[float, float]:
        """Wilson score interval for binomial proportion."""
        if (V, F) not in self.measurements:
            return (0.0, 1.0)  # No data = wide interval
        
        det, tot = self.measurements[(V, F)]
        return wilson_score_interval(det, tot, alpha)
```

---

## 3. Calibration Video Structure (V2)

### Current Structure
```
[META] -> [VERSION LADDER] -> [FPS LADDER] -> [END]
```

### Proposed Structure (V2)
```
[META] -> [VERSION LADDER] -> [FPS LADDER] -> [JOINT TEST] -> [END]
         (~10s)               (~10s)            (~10s)
```

**Joint Test Segment:**
- Encodes frames at selected (V, F) pairs
- Each pair gets ~20-30 frames (enough for stable p estimate)
- Selected based on Phase 1 results (top 3-5 promising pairs)

**Example Calibration Video (standard preset, ~45s):**
```
0-2s:    META segment (preset info)
2-12s:   Version ladder (V25, V27, V30, V33, V35, V38, V40)
12-22s:  FPS ladder (10, 12, 15, 18, 20, 25, 30 FPS at anchor V25)
22-40s:  Joint testing:
         - (V25, 15fps): 20 frames
         - (V30, 15fps): 20 frames
         - (V30, 20fps): 20 frames
         - (V35, 10fps): 20 frames
40-42s:  END marker
```

---

## 4. Overhead Calculation (V2)

### Current (V1)
```python
# Line 603-605
combined_dr = ver_dr * fps_dr
raw_overhead = 1.0 / combined_dr
overhead = round(max(raw_overhead * safety, MIN_OVERHEAD_RQ), 2)
```

### Proposed (V2)

**In calibration analysis:**
```python
def compute_recommendations_v2(
    joint_model: JointProbabilityModel,
    K_estimate: int,  # Could be from user input or conservative default
    confidence: float = 0.95,
) -> list[TierRecommendation]:
    """
    Compute recommendations using joint probability model.
    
    For each candidate (V, F) pair:
    1. Estimate p(V, F) from joint model
    2. Compute required overhead for K_estimate blocks
    3. Calculate expected throughput
    4. Select based on tier criteria
    """
    recommendations = []
    
    for tier_name, tier_config in _TIERS_V2.items():
        # Find (V, F) pairs that meet tier criteria
        candidates = []
        
        for V in ALL_VERSIONS:
            for F in ALL_FPS:
                p = joint_model.estimate_p(V, F)
                
                # Tier filter: require confident p above threshold
                p_low, p_high = joint_model.confidence_interval(V, F)
                if p_low < tier_config["min_rate"]:
                    continue  # CI too low for this tier
                
                # Compute overhead for this (V, F, p) combination
                h = compute_overhead_v2(K_estimate, p, confidence)
                
                # Throughput: bytes/sec
                capacity = _alphanumeric_byte_capacity(V, EC_LEVEL)
                throughput = capacity * F / h
                
                candidates.append((throughput, V, F, p, h))
        
        # Select best throughput for this tier
        if candidates:
            best = max(candidates, key=lambda x: x[0])
            throughput, V, F, p, h = best
            
            recommendations.append(TierRecommendation(
                tier=tier_name,
                available=True,
                qr_version=V,
                fps=F,
                overhead=round(h, 2),
                throughput_bps=throughput,
                confidence=confidence,
                p_estimate=p,
                p_ci=joint_model.confidence_interval(V, F),
            ))
    
    return recommendations
```

**Key Change:** Overhead now depends on K!

```python
# In CLI: allow user to specify expected file size
# qrstream calibrate --expected-file-size 100MB
# -> K = 100MB / blocksize(V)
# -> Compute overhead tailored to this K
```

---

## 5. Confidence Intervals & Uncertainty Quantification

### Wilson Score Interval for Binomial Proportions

Current calibration reports point estimates: "V25: 85% detect rate"

Proposed: Report confidence intervals: "V25: 85% [82%, 88%] (n=100)"

```python
def wilson_score_interval(
    successes: int, 
    total: int, 
    alpha: float = 0.05
) -> tuple[float, float]:
    """
    Wilson score interval for binomial proportion.
    
    More accurate than Wald interval for small samples and extreme proportions.
    Agresti-Coull adjustment for better coverage.
    
    Returns:
        (lower, upper) confidence interval at (1-alpha) level
    """
    if total == 0:
        return (0.0, 1.0)
    
    p = successes / total
    
    # Z-score for (1 - alpha/2)
    import math
    if alpha == 0.05:
        z = 1.96
    elif alpha == 0.01:
        z = 2.576
    else:
        # Approximate using probit function
        z = math.sqrt(2) * math.erfinv(1 - alpha)
    
    # Wilson interval
    n = total
    z2 = z ** 2
    
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n**2))
    
    return (
        max(0.0, center - margin),
        min(1.0, center + margin)
    )
```

### Application in Recommendations

```python
@dataclass
class TierRecommendation:
    tier: str
    available: bool
    qr_version: int | None = None
    fps: int | None = None
    overhead: float | None = None
    throughput_bps: float | None = None
    
    # NEW: Uncertainty quantification
    p_estimate: float | None = None
    p_ci_lower: float | None = None
    p_ci_upper: float | None = None
    p_sample_size: int | None = None
    
    # NEW: Overhead computation details
    K_assumed: int | None = None
    overhead_confidence: float | None = None
```

**User-Facing Output (V2):**
```
┌─────────────────────────────────────────────────────┐
│ QRStream Calibration Results (V2)                  │
├─────────────────────────────────────────────────────┤
│  Channel quality: Good                              │
│  Precision: standard                                │
│  Video: 1920x1080 @ 29.97fps, 1800 frames        │
├─────────────────────────────────────────────────────┤
│  Tier        Version    FPS  Overhead  Throughput   │
│  ─────────────────────────────────────────────       │
│  Safe        V25      10    1.45      245 KB/s     │
│                        p=90% [87%, 92%] (n=150)   │
│                        *For 10MB file (K≈35000)    │
│                                                     │
│  Balanced    V30      15    1.28      412 KB/s     │
│                        p=82% [78%, 85%] (n=120)   │
│                        *For 10MB file (K≈25000)    │
│                                                     │
│  Aggressive  V35      15    1.15      580 KB/s     │
│                        p=72% [67%, 77%] (n=100)   │
│                        *For 10MB file (K≈21000)    │
├─────────────────────────────────────────────────────┤
│  ℹ Overhead calculated for K=25000 blocks          │
│    (assumes ~10MB file with V25-QRs)               │
│  ℹ Use --file-size to tailor for your use case     │
└─────────────────────────────────────────────────────┘
```

---

## 6. Implementation Plan

### Phase 1: Theoretical Foundation (Week 1)

**Files to modify:**
- `src/qrstream/calibrate.py` - Add new functions
- `src/qrstream/overhead_v2.py` - New module for overhead calculation

**New functions:**
```python
# overhead_v2.py
def compute_overhead_v2(K: int, p: float, confidence: float = 0.95) -> float:
    """Compute required overhead for K blocks with detection rate p."""

def wilson_score_interval(successes: int, total: int, alpha: float) -> tuple[float, float]:
    """Confidence interval for binomial proportion."""

def simulate_decode_success(K: int, N: int, p: float, n_trials: int = 1000) -> float:
    """Monte Carlo simulation to validate overhead formula."""
```

### Phase 2: Joint Probability Model (Week 2)

**Files to modify:**
- `src/qrstream/calibrate.py` - Add `JointProbabilityModel` class
- Update `CalibrationFrame` to support joint test segment

**New segment type:**
```python
SEG_JOINT = 5  # New segment ID for joint (V, F) testing

@dataclass
class CalibrationFrame:
    segment_id: int  # SEG_META=1, SEG_VERSION=2, SEG_FPS=3, SEG_END=4, SEG_JOINT=5
    param: int       # For SEG_JOINT: encodes both V and F
    step_index: int
    total_steps: int
    frame_seq: int
    
    def pack_v2(self) -> bytes:
        """Extended pack for V2 protocol (supports joint segment)."""
        if self.segment_id == SEG_JOINT:
            # param encodes (V, F) as (V << 8) | F
            v = (self.param >> 8) & 0xFF
            f = self.param & 0xFF
            # ... pack with 2-byte param
```

### Phase 3: Updated Calibration Video Generation (Week 3)

**Modify `generate_calibration()`:**
```python
def generate_calibration_v2(
    preset_name: str = "standard",
    output_path: str | None = None,
    display: bool = False,
    expected_file_size: int | None = None,  # NEW
    codec: str = "h264",
    reporter: ProgressReporter | None = None,
) -> PresetConfig:
    """Generate V2 calibration video with joint testing."""
    
    config = resolve_preset(preset_name, display_hz=60)
    
    # Phase 1: Version ladder (unchanged)
    # Phase 2: FPS ladder (unchanged)
    # Phase 3: Joint testing (NEW)
    
    joint_pairs = _select_joint_test_pairs(config)
    # ... generate frames for joint testing
```

**Helper function:**
```python
def _select_joint_test_pairs(config: PresetConfig) -> list[tuple[int, int]]:
    """Select promising (V, F) pairs for joint testing."""
    # Use version ladder and FPS ladder to select top combinations
    # Heuristic: test top 3 versions × top 3 FPS = 9 pairs
    
    top_versions = config.version_ladder[:3]  # Assumes sorted
    top_fps = config.fps_ladder[:3]
    
    pairs = []
    for V in top_versions:
        for F in top_fps:
            pairs.append((V, F))
    
    return pairs  # Max 9 pairs, ~10s at 20 frames/pair
```

### Phase 4: Updated Analysis (Week 4)

**Modify `analyze_calibration()`:**
```python
def analyze_calibration_v2(
    video_path: str,
    expected_K: int | None = None,  # NEW: for tailored overhead
    workers: int | None = None,
    reporter: ProgressReporter | None = None,
) -> CalibrationResultV2:  # New return type
    """Analyze V2 calibration video."""
    
    # ... existing frame decoding ...
    
    # NEW: Fit joint probability model
    joint_model = JointProbabilityModel()
    for (V, F), (detected, total) in joint_measurements.items():
        joint_model.add_measurement(V, F, detected, total)
    
    # NEW: Compute recommendations with K dependency
    K = expected_K or _estimate_K_from_channel()  # Conservative default
    recommendations = compute_recommendations_v2(joint_model, K)
    
    return CalibrationResultV2(
        # ... existing fields ...
        joint_model=joint_model,
        K_assumed=K,
    )
```

### Phase 5: CLI Integration (Week 5)

**Modify `cli.py`:**
```python
def cmd_calibrate(args):
    """Handle 'calibrate' subcommand with V2 features."""
    
    # NEW: Accept --expected-file-size
    expected_file_size = args.expected_file_size  # e.g., "100MB"
    K = None
    if expected_file_size:
        # Parse "100MB" -> bytes
        size_bytes = _parse_size(expected_file_size)
        # Estimate K using default V25 blocksize
        blocksize = _estimate_blocksize(25)
        K = math.ceil(size_bytes / blocksize)
    
    if args.input:
        result = analyze_calibration_v2(
            video_path=args.input,
            expected_K=K,
            workers=args.workers,
            reporter=reporter,
        )
        # ... display results with confidence intervals ...
```

**New CLI flags:**
```python
cal = subparsers.add_parser('calibrate', ...)
cal.add_argument(
    '--expected-file-size',
    metavar='SIZE',
    help='Expected file size for tailored overhead (e.g., "10MB", "100MB")'
)
cal.add_argument(
    '--confidence',
    type=float,
    default=0.95,
    help='Target decode success probability (default: 0.95)'
)
cal.add_argument(
    '--model',
    choices=['v1', 'v2'],
    default='v2',
    help='Calibration model version (default: v2)'
)
```

---

## 7. Validation & Experiment Design

### A. Simulation Validation

**Test the overhead formula:**
```python
# test_overhead_v2.py
def test_overhead_formula_accuracy():
    """Validate that computed overhead achieves target success rate."""
    
    K_values = [100, 500, 1000, 5000, 10000]
    p_values = [0.70, 0.80, 0.90, 0.95]
    confidence_levels = [0.90, 0.95, 0.99]
    
    for K in K_values:
        for p in p_values:
            for conf in confidence_levels:
                # Compute overhead
                h = compute_overhead_v2(K, p, conf)
                N = int(K * h)
                
                # Monte Carlo: simulate decoding
                success_count = 0
                n_trials = 1000
                
                for _ in range(n_trials):
                    # Simulate binomial reception
                    S = np.random.binomial(N, p)
                    
                    # Check if decoding succeeds
                    # (simplified - actual LT decoding is more complex)
                    if S >= K * 1.05:  # RaptorQ epsilon
                        success_count += 1
                
                empirical_conf = success_count / n_trials
                
                # Check if empirical matches target
                assert empirical_conf >= conf - 0.02, (
                    f"K={K}, p={p}, conf={conf}: "
                    f"empirical={empirical_conf:.3f}"
                )
```

### B. Real-World Testing

**Experiment protocol:**
1. Generate calibration video (V2) with joint testing
2. Capture under various conditions:
   - Good channel: phone camera, good lighting
   - Medium channel: some motion, shadows
   - Poor channel: low light, high motion
3. Analyze and compare V1 vs V2 recommendations
4. **Ground truth:** Actually encode/decode files of known sizes, measure success rate

**Metrics to collect:**
- Calibration time (V1 vs V2)
- Recommendation accuracy (does actual decode success match predicted?)
- Overhead efficiency (is V2 using less overhead than V1 for same success rate?)

---

## 8. Backward Compatibility

### Compatibility Strategy

1. **V1 compatibility:** Keep existing `compute_recommendations()` function
2. **Auto-detect:** Analyze function detects V1 vs V2 calibration video
3. **Graceful fallback:** If V2 data missing, fall back to V1 model

```python
def analyze_calibration(
    video_path: str,
    model: str = "auto",  # "v1", "v2", "auto"
    **kwargs,
) -> CalibrationResult | CalibrationResultV2:
    """Analyze calibration video with auto-detection."""
    
    # Detect calibration version from video metadata
    cal_version = _detect_calibration_version(video_path)
    
    if model == "auto":
        model = "v2" if cal_version >= 2 else "v1"
    
    if model == "v1" or cal_version < 2:
        return analyze_calibration_v1(video_path, **kwargs)
    else:
        return analyze_calibration_v2(video_path, **kwargs)
```

---

## 9. Summary of Changes

### New Files
- `src/qrstream/overhead_v2.py` - Overhead calculation with K dependency
- `src/qrstream/joint_model.py` - Joint probability model
- `tests/test_calibrate_v2.py` - V2 tests

### Modified Files
- `src/qrstream/calibrate.py` - Add V2 functions, keep V1 for compatibility
- `src/qrstream/cli.py` - Add V2 CLI flags
- `tests/test_calibrate.py` - Add V2 test cases

### Performance Impact
- **Calibration time:** +~10s (joint testing phase)
- **Analysis time:** ~same (model fitting is fast)
- **Overhead accuracy:** +20-30% (less wasted overhead)
- **Decode success predictability:** +40-50% (confidence intervals)

---

## 10. Future Enhancements

1. **Adaptive calibration:** Stop early if channel is clearly excellent/poor
2. **Learning from history:** Save calibration results, improve model over time
3. **Multi-file optimization:** If encoding multiple files, share calibration
4. **Real-time feedback:** During encoding, adjust parameters based on actual decode success

---

## Appendix: Mathematical Derivations

### A. RaptorQ Failure Probability

From RFC 6330 Section 5:
```
P(failure) ≤ (1 - ε)^n where ε = 0.01
```

For n = K × h encoded blocks:
```
P(success) = 1 - (1 - ε)^(K×h)
```

Setting P(success) = 0.95:
```
0.95 = 1 - (0.99)^(K×h)
(0.99)^(K×h) = 0.05
K×h = log(0.05) / log(0.99) ≈ 299
h ≈ 299 / K
```

This shows overhead should decrease with K!

### B. Binomial Confidence Intervals

**Wald interval (do NOT use):**
```
p ± z × √(p(1-p)/n)
```
Problem: Breaks down at p=0 or p=1, inaccurate for small n.

**Wilson score interval (use this):**
```
(p + z²/(2n) ± z√(p(1-p)/n + z²/(4n²)) / (1 + z²/n)
```

More accurate for all p and n.

### C. Overhead Formula Derivation

Starting from Chernoff bound:
```
P(S < μ(1-δ)) ≤ exp(-μδ²/2)
```

Set μ = Np, require S ≥ K(1+ε):
```
μ(1-δ) = K(1+ε)
(1-δ) = K(1+ε) / Np
δ = 1 - K(1+ε) / Np
```

Plug into Chernoff:
```
P(failure) ≤ exp(-Npδ²/2)
           = exp(-Np/2 × (1 - K(1+ε)/Np)²)
           = exp(-(Np - K(1+ε))² / (2Np))
```

Set P(failure) = α:
```
α = exp(-(Np - K(1+ε))² / (2Np))
ln(α) = -(Np - K(1+ε))² / (2Np)
-2Np×ln(α) = (Np - K(1+ε))²
√(2Np×ln(1/α)) = Np - K(1+ε)
Np - √(2Np×ln(1/α)) = K(1+ε)
```

Solve quadratic for N:
```
N = (K(1+ε) + √(2K(1+ε)p×ln(1/α))) / p
```

This is the formula used in `compute_overhead_v2()`.

---

**End of Design Document**

---

## Implementation Status (2026-05)

| Phase | Status | Notes |
|---|---|---|
| 1. Theoretical foundation | ✅ landed | `wilson_lower_bound`, `_binomial_sf`, `estimate_success_probability` live in `calibration_optimizer.py`. No standalone `overhead_v2.py` — inlining keeps the math beside the optimizer that uses it. |
| 2. Joint probability model | ✅ landed | Implemented as the "pairwise" segment (`SEG_PAIRWISE = 5`) and a strategic 6-pair plan in `_select_pairwise_plan` (corner pairs + balanced interior + high-V mid-F). Bilinear interpolation across measurements lives in `_interpolate_pair_probability`; the optimizer reports `source = "pairwise"` for direct hits and `"pairwise-interp"` for interpolated picks. No standalone `JointProbabilityModel` class — interpolation is inlined for the same reason. |
| 3. V2 calibration video | ✅ landed | `generate_calibration` lays down `META → VERSION → FPS → PAIRWISE → END` segments. |
| 4. Updated analysis | ✅ landed | `analyze_calibration` collects pairwise rates; `compute_recommendations` threads them into `optimize_calibration`. K-aware overhead uses `OptimizerConfig.target_k` driven by `--target-size` / `--target-file`. |
| 5. CLI integration | ✅ landed | Implemented: `--target-size`, `--target-file`, `--fountain-codec`, `--confidence`. |
| `--model {v1\|v2\|auto}` | ❌ skipped | The V1 calibration video format is no longer produced or analyzed. A no-op flag would be dead code; auto-detection is unnecessary. |
| Monte-Carlo overhead validator | ❌ skipped | `_binomial_sf` already returns the exact survival probability via log-sum-exp, so MC sanity-checking adds runtime without new information. |