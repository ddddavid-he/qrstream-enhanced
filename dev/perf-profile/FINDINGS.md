# Profiling Findings

Measured on **macOS (Apple Silicon, 14-core, Python 3.13, OpenCV 4.13.0)**
and cross-verified in a Linux/aarch64 podman container (OpenCV 4.13.0).
Both environments show identical behaviour for the core findings below.

---

## Headline: `generate_qr_image` never actually uses OpenCV's fast encoder

100% of encode frames fall through to the slow `qrcode` Python library.
This happens because of a capacity mismatch between the **ISO 18004
standard table** (which `auto_blocksize` uses) and **OpenCV's
`QRCodeEncoder` byte-mode actual capacity** (which is ~68% of ISO).

### Measured OpenCV 4.13 byte-mode capacity (high-entropy bytes)

| QR version | ISO 18004 (M) | OpenCV byte-mode (M) | Ratio |
|------------|--------------:|---------------------:|------:|
| V10/M      | 271 B         | 143 B                | 53%   |
| V15/M      | 535 B         | 271 B                | 51%   |
| V20/M      | 666 B         | **451 B**            | 68%   |
| V25/M      | 1091 B        | 662 B                | 61%   |
| V30/M      | 1624 B        | 916 B                | 56%   |
| V40/M      | 2331 B        | 1556 B               | 67%   |

Our default setup (V20/M, binary_qr, V3 protocol):
- `auto_blocksize` → `blocksize = 635` (from ISO 666 − header 28 − COBS 3)
- packed block = 635 + 28 = 663 B
- after COBS = ~666 B
- `cv2.QRCodeEncoder.encode("V20", mode=BYTE)` → **raises `cv2.error`**
  ("given version is not suitable for the given input string length")
- `except (RuntimeError, cv2.error)` → falls back to `qrcode` library
- user sees `version=20` succeed but **never notices the fallback**

Evidence from single-process `cProfile` on a 100 KB encode (26.5s total):

```
   ncalls  tottime  percall  filename:lineno(function)
     2925    5.623    0.002  main.py:474(map_data)           ← qrcode lib
     2600    3.526    0.001  util.py:270(_lost_point_level3) ← qrcode lib
      324    2.104    0.006  cv2.VideoWriter.write
     2600    2.002    0.001  util.py:200(_lost_point_level1) ← qrcode lib
336588/7013  1.751    0.000  base.py:274(__mod__)            ← qrcode lib
     2600    1.267    0.000  util.py:243(_lost_point_level2) ← qrcode lib
```

Top 6 functions = 16.3s / 26.5s = **62% of encode time is in the
`qrcode` fallback library**. OpenCV's fast path contributes exactly zero.

---

## But here's the twist: fixing the fallback does NOT speed things up

Running both encoders head-to-head on 324 realistic V3 binary blocks:

| Encoder                                          | time/frame |
|--------------------------------------------------|-----------:|
| OpenCV V20 byte-mode (blocksize=380, fits capacity) | 41.86 ms/frame |
| `qrcode` lib V20 (current fallback, blocksize=635)  | 41.06 ms/frame |

They are **essentially the same speed**. OpenCV's Python-binding
`QRCodeEncoder` is not actually fast.

So the fallback isn't a performance bug. It IS still two other bugs:

1. **UX bug**: the user asks for V20 but silently gets a larger QR
   (qrcode library auto-upgrades to V21+V22 via `fit=True` when the
   payload doesn't fit). Frame size, video file size, and detection
   latency all grow without the user knowing.
2. **Truth-in-advertising bug**: `qr_utils.py` docstring says
   "Default: OpenCV QRCodeEncoder (much faster)" — this is never true
   for binary_qr mode.

---

## Where the encode time actually goes (macOS, 14 workers, 100 KB file)

Staged multi-process timing:

```
 wall=4.255s  frames=324  K=162  blocksize=635  frame=1250x1250
  pool.map (QR gen + IPC)   :  2.088s  (49.1%)   ← qrcode lib + IPC
  VideoWriter.write         :  2.086s  (49.0%)   ← mp4v muxing
  cv2.resize (if mismatch)  :  0.024s  ( 0.6%)
  other                     :  0.057s  ( 1.3%)
  throughput                :  76.1 frames/s
```

QR generation and video muxing are **neck and neck** at ~49% each.

Extrapolated wall time (workers=14, macOS):
- 100 KB → 4.3 s (measured)
- 1 MB → ~43 s
- 10 MB → ~7 min (matches user's "超过10MB时间非常长")

---

## Hot-path micro-benchmarks (macOS, median per call)

| Operation                          | median    | notes                       |
|------------------------------------|----------:|-----------------------------|
| `generate_qr_image` V20 (345 B)    | 40.9 ms   | via `qrcode` fallback       |
| `generate_qr_image` V25 (510 B)    | 60.3 ms   |                             |
| `generate_qr_image` V40 (1174 B)   | 142.0 ms  |                             |
| `try_decode_qr` V20 (1090²)        | 102.9 ms* | *first-call / WeChatQR init |
| `try_decode_qr` V25 (1290²)        | 7.3 ms    |                             |
| `try_decode_qr` V40 (1890²)        | 16.5 ms   |                             |
| `LTEncoder.generate_block` K=1000  | 3.9 µs    | negligible                  |
| `BlockGraph` full decode K=5000    | 77 ms total (15 µs/block) | not a bottleneck |
| `cobs_encode` 1000 B               | 55.7 µs   | not a bottleneck            |
| `unpack` V3 with CRC               | 0.8 µs    | not a bottleneck            |
| `cv2.imencode` JPEG q95 540²       | 1.2 ms    |                             |
| `cv2.imencode` JPEG q75 540²       | 0.7 ms    | **-40% time, -49% payload** |
| `cv2.imdecode` JPEG q95 540²       | 1.6 ms    |                             |
| `cv2.imdecode` JPEG q75 540²       | 1.2 ms    | **-29% time**               |
| `cv2.VideoWriter.write` mp4v 1290² | 1.5 ms    |                             |

---

## Staged decode (macOS, 14 workers)

| size  | wall  | probe | frame_read | worker | LT decode |
|-------|------:|------:|-----------:|-------:|----------:|
| 10 KB | 0.61 s| 100%  | —          | —      | early-term during probe |
| 100 KB| 2.68 s| 71.6% | 11.9%      | 14.1%  | 0.1%       |

Decode is probe-dominated for small files. The probe phase reads
3 windows × 120 frames = 360 frames worth of QR detection, and small
videos have early termination before the main scan starts.

---

## Previously-suggested optimisations — final verdict

| Hypothesis                              | Verdict | Evidence                                  |
|-----------------------------------------|---------|-------------------------------------------|
| COBS is a bottleneck                    | ❌ FALSE | 56 µs/1 KB, <0.2% of per-frame time       |
| IPC JPEG transfer is a major cost       | ⚠️ MINOR | 14% of decode wall at 100 KB              |
| Batch `generate_blocks`                 | ❌ FALSE | 4 µs/block, irrelevant                    |
| PRNG sampling for large d               | ❌ FALSE | 1 µs, d≤30                                |
| `_to_np` type check overhead            | ❌ FALSE | ns-level                                  |
| Adaptive frame skipping                 | ✅ WORKING | sample_rate already auto                 |
| libx264 codec                           | ❌ DANGEROUS | would destroy QR fidelity             |
| cProfile-guided optimisation            | ✅ CORRECT | proved its value here                    |
| JPEG quality 95 → 75                    | ✅ WORTHWHILE | -30% encode, -50% payload              |
| Worker-side seek instead of IPC         | ❌ NOT WORTH IT | IPC only 14% of decode             |
| In-place XOR in `generate_block`        | ❌ FALSE | generate_block is 4 µs, not hot          |

---

## What to actually fix (ranked by value)

### 1. The `generate_qr_image` OpenCV-encoder bug (UX, not perf)

The fallback silently upgrades the user's requested version. Three
possible fixes:

- **(a) Remove the OpenCV fast path for binary mode entirely.** It
  was never fast anyway. Use `qrcode` unconditionally. Simplest.
  Side effect: `version=20` actually stays at V20 only if the data
  fits; otherwise `qrcode` library auto-upgrades via `fit=True`.
- **(b) Fix `auto_blocksize` to use OpenCV's real byte-mode capacity
  table** (the numbers above) instead of ISO. Then OpenCV always
  succeeds at the requested version, matching user expectation.
  Cost: ~30% less user payload per frame → ~40% more frames → ~40%
  longer video. Users see no surprise in version, but encode time
  gets worse.
- **(c) Keep a dual table: "capacity for OpenCV" and "capacity for
  qrcode".** In base64 mode use OpenCV-capacity (OpenCV is faster
  there); in binary mode just use qrcode and the ISO capacity.
  Most correct, most work.

**Recommended: (a) for now.** Simplest, no surprise for users, no
measurable perf loss. Revisit when OpenCV's byte-mode encoder is fixed
upstream.

### 2. Run `VideoWriter.write` in a separate thread

Currently: `pool.map(generate_qr_image, batch)` → then the main
thread serialises `writer.write(qr_img)` for the whole batch. That
means pool workers are idle while the main thread writes. Split into
a writer thread with a bounded queue: expected ~30-40% encode speedup.

### 3. Drop default JPEG quality in decoder from 95 → 75

One-line change. Saves ~30% of `imencode`+`imdecode` time and ~50%
of IPC payload. Needs a verification test (full encode-decode roundtrip
with various inputs) to confirm no detection loss.

### 4. Reduce probe window for small videos

`PROBE_WINDOW_SIZE=120` and 3 windows = 360 probe frames. For videos
where `total_frames < 1000`, probing reads the whole video. Consider:
- Skip probe if `total_frames < 2 × PROBE_WINDOW_SIZE` and just use
  `sample_rate=1`.
- Or shrink `PROBE_WINDOW_SIZE` when total is small.

---

## Things that are NOT worth pursuing for ≤10 MB

- Rewriting COBS in Cython
- numpy-vectorising `BlockGraph`
- Shared-memory frame transfer
- Rewriting PRNG
- Batching `generate_blocks`

All rounding errors.
