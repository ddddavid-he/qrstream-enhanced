# Discovery — Decode Performance Optimizations After zxing-cpp

**Date**: 2026-05-07  
**Branch**: `dev`  
**Author**: benchmark / handoff note for follow-up implementation agent

---

## Scope

This note records the decode-side performance experiments run after the
`zxing-cpp` migration landed on `dev`.

The goal was not to re-benchmark detector quality; that work had already been
completed on the feature branch that introduced `zxing-cpp`. The goal
here is to identify which decode-pipeline optimizations are still worth
implementing now that:

1. QR detection is `zxing-cpp` instead of WeChatQRCode.
2. Detection runs inside a `ThreadPoolExecutor`.
3. The hot path has shifted away from detector instability and toward frame
   preparation / video reading / probe orchestration.

---

## Current Pipeline Summary

The current decode path in [`src/qrstream/decoder.py`](../src/qrstream/decoder.py)
works like this:

1. **Probe**
   - Phase 1 reads short bursts at high resolution to derive `crop_box`.
   - Phase 2 runs a multi-resolution sweep to learn `adaptive_max_dim`.
   - Phase 3 reads three larger windows and estimates `sample_rate`.

2. **Main scan**
   - `_read_frames()` opens `cv2.VideoCapture`, reads sequentially, applies
     `crop -> downscale -> contiguous copy`, then yields `(frame_idx, frame)`.
   - `_prefetch_iter()` moves read/prepare work onto a producer thread.
   - `_stream_scan()` keeps `workers * 2` detection tasks in flight.

3. **Recovery**
   - `_read_frame_ranges()` re-opens the video and seeks to selected ranges.
   - Recovery levels reuse the same frame-prep pipeline.

Relevant code points:

- [`src/qrstream/decoder.py`](../src/qrstream/decoder.py#L386) `_downscale_frame()`
- [`src/qrstream/decoder.py`](../src/qrstream/decoder.py#L421) `_crop_frame()`
- [`src/qrstream/decoder.py`](../src/qrstream/decoder.py#L437) `_prepare_frame()`
- [`src/qrstream/decoder.py`](../src/qrstream/decoder.py#L913) `_read_frames()`
- [`src/qrstream/decoder.py`](../src/qrstream/decoder.py#L948) `_read_frame_ranges()`
- [`src/qrstream/decoder.py`](../src/qrstream/decoder.py#L1116) `_probe_sample_rate()`
- [`src/qrstream/decoder.py`](../src/qrstream/decoder.py#L2112) `_stream_scan()`

Important implementation detail:

- `_crop_frame()` is cheap: it returns a view.
- `_downscale_frame()` uses `cv2.resize(..., INTER_AREA)`.
- `_prepare_frame()` currently always does `np.ascontiguousarray(...).copy()`
  even when `cv2.resize()` already returned a fresh contiguous array.

---

## Benchmark Setup

### Assets

- `~/Downloads/testcase/IMG_9442.MOV` + `9442.bin`
- `~/Downloads/testcase/IMG_9455.MOV` + `9455.bin`

These are short real-phone videos and are good enough for fast iteration.

### Environments used

1. **Podman / Fedora test container**
   - Used for thread-pool vs process-pool comparison on Linux.
2. **Local macOS host**
   - Used for reader / pipeline optimization experiments on the existing dev
     checkout.

### Validation rule

Every variant had to:

1. Return the same recovered block count as baseline.
2. Reconstruct the file successfully.
3. Match the reference SHA-256.

---

## Baseline Findings

### 1. ThreadPool vs ProcessPool after zxing-cpp

#### Raw detection benchmark inside Podman

Prepared-frame benchmark on `IMG_9442.MOV` (`204` sampled frames):

| workers | ThreadPool | ProcessPool | Process / Thread |
|---|---:|---:|---:|
| 1 | 0.5021s | 2.1986s | 4.38x slower |
| 2 | 0.2526s | 1.5042s | 5.95x slower |
| 4 | 0.1347s | 1.5707s | 11.66x slower |
| 6 | 0.0922s | 1.6271s | 17.64x slower |

Prepared-frame benchmark on `IMG_9455.MOV` (`152` sampled frames):

| workers | ThreadPool | ProcessPool | Process / Thread |
|---|---:|---:|---:|
| 1 | 0.3268s | 1.4723s | 4.51x slower |
| 4 | 0.0905s | 1.0846s | 11.99x slower |
| 6 | 0.0630s | 1.1458s | 18.20x slower |

Interpretation:

- For **pure zxing-cpp detection**, `ThreadPoolExecutor` is decisively better.
- The old `ProcessPool` argument mostly disappeared with WeChatQRCode's removal:
  there is no crash isolation need here, and the IPC cost dominates.

#### Full decode benchmark inside Podman

Full `extract_qr_from_video -> decode_blocks` benchmark with `workers=4`:

| video | ThreadPool total | ProcessPool total | Result |
|---|---:|---:|---|
| `IMG_9442.MOV` | 29.883s | 29.230s | near parity, process ~2% faster |
| `IMG_9455.MOV` | 34.349s | 36.512s | thread ~6% faster |

Interpretation:

- End-to-end decode is now much more **reader/probe bound** than detector bound.
- ProcessPool is not delivering a clear full-pipeline win.
- Given the much worse pure-detect numbers and the extra complexity,
  **do not bring ProcessPool back**.

### 2. Legacy profile data is now only directional

[`dev/perf-profile/results/decode_report.txt`](./perf-profile/results/decode_report.txt)
was collected before the zxing-cpp migration and still contains JPEG IPC /
ProcessPool-era costs.

It is still useful for one broad point:

- LT decoding itself is tiny.
- Video read / frame prep / probe orchestration dominate.

But the exact percentages in that file should **not** be treated as current
zxing-cpp numbers.

---

## Optimization Experiments

The list below uses the original priority labels from discussion, then records
what the benchmarks actually showed.

### A. Highest Priority Candidate — eliminate avoidable extra frame copy

Prototype:

- Patch `_prepare_frame()` so that if `cv2.resize()` already returned a fresh,
  contiguous array not sharing memory with the source frame, we return it
  directly instead of doing another `.copy()`.

Reasoning:

- Crop is a view, so it still needs a copy later.
- But resize usually allocates a new output buffer anyway.
- Current code may pay for a redundant full-frame copy after resize.

Measured result:

| video | baseline | patched | delta |
|---|---:|---:|---:|
| `IMG_9442.MOV` | 14.892s | 14.735s | `+1.06%` |
| `IMG_9455.MOV` | 18.565s | 20.495s | `-10.40%` |

Interpretation:

- The signal is weak and noisy.
- On one clip the patch helped by about `1%`; on another run it regressed.
- This is **not currently strong enough** to justify being the first
  implementation target without a more rigorous repeated benchmark.

Recommendation:

- Keep as a small, low-risk micro-optimization candidate.
- Do not treat it as the primary perf lever.

### B. High Priority Candidate — avoid probe/main overlap re-read

Prototype:

- Record the Phase-3 probe window ranges.
- During main scan, skip frames whose indices already belong to those ranges.

Reasoning:

- Probe already decoded those windows.
- Main scan rereads them today.
- In theory this should remove duplicated I/O and duplicated frame prep.

Measured result on `IMG_9442.MOV`:

| baseline | patched | delta |
|---:|---:|---:|
| 14.892s | 14.842s | `+0.34%` |

Interpretation:

- The gain is effectively noise-level.
- The three probe windows are too small to matter much on these short clips.

Recommendation:

- Do not prioritize this as an isolated optimization.
- If touched later, it should be part of a larger probe/main unification,
  not a standalone patch.

### C. Medium Priority Candidate — cache prepared probe frames

Prototype:

- Cache frames emitted by `_read_frame_ranges()` during probe using key
  `(frame_idx, max_detect_dim, crop_box)`.
- Let `_read_frames()` reuse cached prepared frames when the same frame and
  same prep parameters reappear in main scan.

Measured result on `IMG_9442.MOV`:

Cold-first run:

| baseline | cached | apparent delta |
|---:|---:|---:|
| 17.688s | 14.960s | `+15.42%` |

Warm-cache sanity check:

| warm baseline | cached | delta |
|---:|---:|---:|
| 14.726s | 14.960s | `-1.59%` |

Interpretation:

- The apparent `15%` win was mostly a filesystem cache artifact.
- Once baseline was rerun warm, frame caching no longer helped.
- For these clips, the overlap set is too small to justify the extra caching
  complexity and memory retention.

Recommendation:

- **Do not implement this now**.
- If revisited, benchmark again on much longer clips where probe windows are a
  negligible fraction of total work but recovery may benefit from reusing exact
  prepared frames.

### D. Medium Priority Candidate — replace OpenCV reader with PyAV

Prototype:

- Monkey-patch `_read_frames()` and `_read_frame_ranges()` to decode frames via
  `PyAV`, then pass `frame.to_ndarray(format='bgr24')` through the existing
  `_prepare_frame()` pipeline.

Measured result on `IMG_9442.MOV`:

| baseline | PyAV reader | delta |
|---:|---:|---:|
| 14.645s | 12.058s | `+17.67%` |

Correctness:

- Same recovered block count (`82`)
- Decode succeeded
- SHA matched reference

Interpretation:

- This is the **largest reliable gain** found in this round.
- It outperformed every other tested optimization by a large margin.
- The original “medium priority” label is no longer justified.

Recommendation:

- **Promote this to the first implementation track.**

Notes for implementation:

- The prototype used naive sequential decode for both full scan and range reads.
- Even with that simplistic implementation, it beat `cv2.VideoCapture`.
- A production implementation should wrap the reader behind a small internal
  abstraction so the fallback path can still use OpenCV when `av` is not
  available or when a platform-specific bug appears.

### E. Medium Priority Candidate — replace reader with ffmpeg rawvideo pipe

Prototype:

- Spawn `ffmpeg` and stream full-resolution `bgr24` frames over stdout.
- Convert bytes to `np.ndarray`, then run the existing `_prepare_frame()` path.

Measured result on `IMG_9442.MOV`:

| baseline | ffmpeg pipe | delta |
|---:|---:|---:|
| 14.645s | 22.779s | `-55.54%` |

Correctness:

- Same recovered block count (`82`)
- Decode succeeded
- SHA matched reference

Interpretation:

- This is not competitive.
- The rawvideo pipe pushes too much memory bandwidth and copy traffic.
- It also provides no natural advantage for the current probe / range-read
  structure.

Recommendation:

- **Do not pursue the ffmpeg-stdout rawvideo design.**

---

## Re-ranked Priority After Measurement

The original priority labels should be updated as follows:

1. **P0 / implement first: PyAV reader backend**
   - Only candidate with a clear, meaningful, correctness-preserving win.
   - Measured improvement: about `18%` on `IMG_9442.MOV`.

2. **P1 / optional micro-optimization: copy elision in `_prepare_frame()`**
   - Very small upside at best.
   - Needs repeated benchmarking before merge.

3. **P2 / probably not worth isolated implementation: probe/main overlap skip**
   - Measured gain only about `0.3%`.

4. **P2 / deprioritize: frame cache of prepared probe frames**
   - No reliable warm-run gain.

5. **Reject: ffmpeg rawvideo pipe reader**
   - Large regression.

6. **Reject: ProcessPool comeback**
   - Pure detect throughput is dramatically worse.
   - Full-pipeline result is near-parity at best, with much higher complexity.

---

## Suggested Implementation Plan For New Agent

### Track 1 — Productize PyAV reader

Target files:

- [`src/qrstream/decoder.py`](../src/qrstream/decoder.py)
- [`pyproject.toml`](../pyproject.toml)
- tests under [`tests/`](../tests)

Suggested shape:

1. Add a small internal reader abstraction, not a public API.
2. Implement a `PyAV`-backed sequential reader for `_read_frames()`.
3. Implement a `PyAV`-backed range reader for `_read_frame_ranges()`.
4. Keep an OpenCV fallback path.
5. Add correctness regression tests on real fixtures plus at least one synthetic
   encode/decode roundtrip.
6. Add a benchmark script or documented command so the gain can be re-verified.

Acceptance criteria:

1. `extract_qr_from_video()` correctness unchanged.
2. No public CLI/API churn required.
3. `IMG_9442.MOV` and `IMG_9455.MOV` still reconstruct correctly.
4. Reader swap preserves crop / adaptive downscale / recovery behavior.

### Track 2 — Optional follow-up micro-bench work

Only after PyAV lands:

1. Re-benchmark `_prepare_frame()` copy elision with repeated runs.
2. If the win remains below noise, drop it.
3. Do not spend time on frame caching or ffmpeg pipe unless a new workload shows
   a very different profile.

---

## Bottom Line

After zxing-cpp, the decode path is no longer detector-bound in the way it was
under WeChatQRCode.

The only optimization from this round that clearly moved end-to-end decode time
is:

- **Replace `cv2.VideoCapture`-based reading with a `PyAV` reader backend.**

Everything else tested in this pass was either noise-level or a regression.
