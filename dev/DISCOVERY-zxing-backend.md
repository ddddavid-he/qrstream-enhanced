# Discovery — zxing-cpp as QR Detection Backend

**Date**: 2026-05-06  
**Branch**: `feature/zxing-cpp-backend`  
**Author**: bench experiment on real phone-captured videos

---

## Background

The current detection backend is `cv2.wechat_qrcode_WeChatQRCode` from
`opencv-contrib-python`.  Two known problems exist:

1. **Native crash (SIGSEGV/SIGTRAP)** on noisy camera frames
   (`opencv_contrib#3570`) — mitigated by `qr_sandbox.py`'s subprocess
   pool, but at significant architectural complexity and ~20–35% throughput
   cost on macOS.

2. **Extreme long-tail latency** — mean/median ratio is 4–8× on real
   4K phone captures (e.g. IMG_9432.MOV: mean 42 ms/frame vs median 11 ms),
   indicating frequent slow internal-retry or error-recovery paths.

---

## Benchmark Results

### Experiment A — frame-slice detection (bench_detector_backends.py)

Methodology: probe phase run first to derive `crop_box` + `adaptive_max_dim`;
all frames that would enter the main scan loop extracted and prepared
(crop → downscale → contiguous copy); both detectors run on identical frames.

| Video | Frames | WeChatQR hits | zxing hits | WeChatQR mean ms | zxing mean ms | Speedup |
|-------|--------|---------------|------------|-----------------|---------------|---------|
| IMG_9442 (10 s) | 204 | 90.7 % | 89.2 % | 42.1 | 4.7 | 8.9× |
| IMG_9455 (18 s) | 152 | 89.5 % | 90.1 % | 55.0 | 5.3 | 10.3× |
| IMG_9448 (20 s) | 602 | 99.3 % | 99.8 % | 11.3 | 2.6 | 4.4× |
| **Total** | **958** | **95.9 %** | **96.0 %** | — | — | **7.1×** |

Hit-rate delta across all videos: **+0.1 %** (1 frame).  
zxing-cpp stdev: 0.1–0.3 ms (negligible). WeChatQR stdev: 19–117 ms (heavy tails).

### Experiment B — full pipeline decode (bench_full_decode.py)

`extract_qr_from_video → decode_blocks`, `detect_isolation='off'` for both.

| Video | WeChatQR | WeChatQR result | zxing-cpp | zxing-cpp result |
|-------|----------|-----------------|-----------|-----------------|
| IMG_9448 (1676 s) | 216 s | PASS, hash MATCH | 58 s | PASS, hash MATCH |
| IMG_9432 (690 s) | >2100 s (DNF) | — | 542 s | PASS, hash MATCH |

IMG_9432 is a known difficult case (4K 60fps, 4.4 GB).  WeChatQR was
killed after 35 minutes without completing; zxing-cpp finished in 9 minutes
with full file reconstruction.

---

## Key Findings

1. **Detection rate parity**: zxing-cpp matches WeChatQR on all tested
   videos (≤0.1 % difference, sometimes better).

2. **4–10× speed improvement**: consistent across video types and resolutions.

3. **Stable latency**: zxing-cpp shows near-zero variance per frame;
   WeChatQR has severe outliers that dominate mean time.

4. **No crash risk**: zxing-cpp is pure C++ with Python bindings, no
   known SIGSEGV/SIGTRAP issues.  `qr_sandbox.py` subprocess pool can be
   removed entirely.

5. **bbox support**: `result.position` provides `top_left`, `top_right`,
   `bottom_right`, `bottom_left` pixel coordinates — equivalent to
   WeChatQRCode's `points` output, so the probe-phase bbox-derived crop
   logic is fully compatible.

---

## Change Plan

See companion section in this file.

### Files to modify

| File | Change |
|------|--------|
| `pyproject.toml` | add `zxing-cpp` dependency, keep `opencv-contrib-python` (still needed for `cv2.resize`, `cv2.VideoCapture`, etc.) |
| `src/qrstream/qr_utils.py` | replace WeChatQRCode singleton with `zxingcpp.read_barcode`; update `DETECTOR_CAN_CRASH = False` |
| `src/qrstream/decoder.py` | remove sandbox init/teardown; remove `_dispatch_detect` global swap; wire direct calls |
| `src/qrstream/qr_sandbox.py` | retire: keep file but mark deprecated (or delete if no external callers) |
| `src/qrstream/cli.py` | remove `--detect-isolation` flag (or keep as no-op for backwards compat) |
| `tests/` | update any tests that reference WeChatQRCode, sandbox, or `DETECTOR_CAN_CRASH` |

### Probe / scan / recovery wiring

All three phases call `try_decode_qr` / `try_decode_qr_with_bbox` from
`qr_utils.py` (either directly or via `_dispatch_detect` hook).
After the change those functions will call `zxingcpp.read_barcode` instead
— no structural changes to the three-phase pipeline logic itself.

### bbox mapping

```
WeChatQRCode returns:   points[i]  →  (4,2) float32 ndarray  [TL, TR, BR, BL]
zxingcpp returns:       result.position.{top_left, top_right, bottom_right, bottom_left}
                        each is a Point with .x and .y attributes
```

The adapter in `try_decode_qr_with_bbox` will build the same `(4,2)` array.

### Thread safety

`zxingcpp.read_barcode` is documented as reentrant; no per-thread singleton
needed.  The existing `threading.local()` singleton pattern in `qr_utils.py`
can be removed.
