#!/usr/bin/env python3
"""
Real-frame benchmark: extract frames from phone-recorded test videos
and measure single-frame + batch detection latency.

Backends tested:
  - OpenCV WeChatQRCode (production baseline)
  - MNN SSD detect (CPU)
  - MNN SSD detect (METAL)
  - MNN SSD detect batch (CPU)
  - MNN SSD detect batch (METAL)
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES = PROJ / "tests" / "fixtures"
MNN_DIR = PROJ / "dev" / "wechatqrcode-mnn-poc" / "models" / "mnn"
CAFFE_DIR = PROJ / "dev" / "wechatqrcode-mnn-poc" / "models" / "caffe"
OUT_DIR = PROJ / "dev" / "wechatqrcode-mnn-poc" / "results"

SSD_TARGET_AREA = 400.0 * 400.0
MAX_DET_DIM = 1080  # match decoder.py
WARMUP = 3
ITERS = 50  # per-frame iterations for latency
BATCH_SIZES = [1, 2, 4, 8, 16]


# ── Frame extraction ─────────────────────────────────────────────

def extract_sample_frames(video_path: str, count: int = 10) -> list[np.ndarray]:
    """Extract evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, count, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            # Downscale like decoder.py
            h, w = frame.shape[:2]
            mx = max(h, w)
            if mx > MAX_DET_DIM:
                s = MAX_DET_DIM / mx
                frame = cv2.resize(frame, (int(w * s), int(h * s)),
                                   interpolation=cv2.INTER_AREA)
            frames.append(frame)
    cap.release()
    return frames


def preprocess_gray(frame: np.ndarray):
    """Preprocess: grayscale + dynamic resize for SSD."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, (SSD_TARGET_AREA / (w * h)) ** 0.5)
    dw, dh = int(w * scale), int(h * scale)
    resized = cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_CUBIC)
    return resized, dw, dh


# ── Benchmarks ───────────────────────────────────────────────────

def bench_wechat(frames: list[np.ndarray], iters: int) -> list[float]:
    det = cv2.wechat_qrcode_WeChatQRCode()
    for f in frames[:WARMUP]:
        det.detectAndDecode(f)
    times = []
    for _ in range(iters):
        for f in frames:
            t0 = time.perf_counter()
            det.detectAndDecode(f)
            times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_mnn_single(frames: list[np.ndarray], iters: int,
                     backend: str = "CPU") -> list[float]:
    import MNN
    interp = MNN.Interpreter(str(MNN_DIR / "detect.mnn"))
    session = interp.createSession({"backend": backend})
    inp = interp.getSessionInput(session)

    # Warmup with first frame
    r, dw, dh = preprocess_gray(frames[0])
    interp.resizeTensor(inp, (1, 1, dh, dw))
    interp.resizeSession(session)
    data = (r.astype(np.float32) / 255.0).reshape(1, 1, dh, dw)
    for _ in range(WARMUP):
        tmp = MNN.Tensor((1, 1, dh, dw), MNN.Halide_Type_Float,
                         data, MNN.Tensor_DimensionType_Caffe)
        inp.copyFrom(tmp)
        interp.runSession(session)

    times = []
    for _ in range(iters):
        for f in frames:
            r, dw, dh = preprocess_gray(f)
            data = (r.astype(np.float32) / 255.0).reshape(1, 1, dh, dw)
            interp.resizeTensor(inp, (1, 1, dh, dw))
            interp.resizeSession(session)
            tmp = MNN.Tensor((1, 1, dh, dw), MNN.Halide_Type_Float,
                             data, MNN.Tensor_DimensionType_Caffe)
            t0 = time.perf_counter()
            inp.copyFrom(tmp)
            interp.runSession(session)
            # Read output to ensure completion
            out = interp.getSessionOutput(session, "detection_output")
            s = out.getShape()
            if s[2] > 0:
                to = MNN.Tensor(s, MNN.Halide_Type_Float,
                                np.zeros(s, dtype=np.float32),
                                MNN.Tensor_DimensionType_Caffe)
                out.copyToHostTensor(to)
            times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_mnn_batch(frames: list[np.ndarray], batch_size: int,
                    iters: int, backend: str = "CPU") -> list[float]:
    """Batch inference: stack N frames and run as batch=N."""
    import MNN
    interp = MNN.Interpreter(str(MNN_DIR / "detect.mnn"))
    session = interp.createSession({"backend": backend})
    inp = interp.getSessionInput(session)

    # All frames must be same size for batch — use the common size
    # Preprocess all frames to a fixed size (use first frame's size)
    processed = []
    for f in frames:
        r, dw, dh = preprocess_gray(f)
        processed.append((r, dw, dh))

    # Use the most common size
    sizes = [(dw, dh) for _, dw, dh in processed]
    common_w, common_h = max(set(sizes), key=sizes.count)

    # Re-resize all to common size
    normed = []
    for r, dw, dh in processed:
        if (dw, dh) != (common_w, common_h):
            r = cv2.resize(r, (common_w, common_h), interpolation=cv2.INTER_CUBIC)
        normed.append(r.astype(np.float32) / 255.0)

    # Warmup
    interp.resizeTensor(inp, (batch_size, 1, common_h, common_w))
    interp.resizeSession(session)
    batch_data = np.stack([normed[i % len(normed)] for i in range(batch_size)])
    batch_data = batch_data.reshape(batch_size, 1, common_h, common_w)
    for _ in range(WARMUP):
        tmp = MNN.Tensor((batch_size, 1, common_h, common_w),
                         MNN.Halide_Type_Float, batch_data,
                         MNN.Tensor_DimensionType_Caffe)
        inp.copyFrom(tmp)
        interp.runSession(session)

    # Benchmark
    times = []
    n_frames = len(normed)
    for _ in range(iters):
        for start in range(0, n_frames, batch_size):
            chunk = [normed[i % n_frames]
                     for i in range(start, start + batch_size)]
            batch_data = np.stack(chunk).reshape(
                batch_size, 1, common_h, common_w)
            tmp = MNN.Tensor((batch_size, 1, common_h, common_w),
                             MNN.Halide_Type_Float, batch_data,
                             MNN.Tensor_DimensionType_Caffe)
            t0 = time.perf_counter()
            inp.copyFrom(tmp)
            interp.runSession(session)
            out = interp.getSessionOutput(session, "detection_output")
            s = out.getShape()
            if s[2] > 0:
                to = MNN.Tensor(s, MNN.Halide_Type_Float,
                                np.zeros(s, dtype=np.float32),
                                MNN.Tensor_DimensionType_Caffe)
                out.copyToHostTensor(to)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed / batch_size)  # per-frame latency
    return times


# ── Stats & printing ─────────────────────────────────────────────

def stats(times: list[float]) -> dict:
    if not times:
        return {}
    a = np.array(times)
    return {"p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
            "mean": round(float(np.mean(a)), 3),
            "min": round(float(np.min(a)), 3),
            "max": round(float(np.max(a)), 3),
            "n": len(times)}


def pr(label: str, times: list[float], base_p50: float = 0):
    s = stats(times)
    if not s:
        print(f"  {label:40s}  (skipped)")
        return s
    sp = ""
    if base_p50 > 0 and s["p50"] > 0:
        sp = f"  {base_p50 / s['p50']:6.1f}x"
    print(f"  {label:40s}  P50={s['p50']:8.2f}ms  P95={s['p95']:8.2f}ms  "
          f"mean={s['mean']:8.2f}ms{sp}")
    return s


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("Real-Frame Pipeline Benchmark — Apple M4 Pro")
    print("=" * 78)
    print(f"Platform : {platform.platform()}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"OpenCV   : {cv2.__version__}")
    try:
        import MNN; print(f"MNN      : {MNN.version()}")
    except ImportError:
        print("MNN      : NOT INSTALLED"); return
    print(f"Iters    : {ITERS} per frame")
    print(f"Batch    : {BATCH_SIZES}")
    print()

    videos = sorted(FIXTURES.glob("**/*.mp4"))
    if not videos:
        print("No test videos found"); return

    all_results = {}

    for vpath in videos:
        tag = f"{vpath.parent.name}/{vpath.name}"
        frames = extract_sample_frames(str(vpath), count=10)
        if not frames:
            print(f"  {tag}: no frames extracted, skip"); continue

        h, w = frames[0].shape[:2]
        print("-" * 78)
        print(f"Video: {tag}  ({w}×{h}, {len(frames)} sample frames)")
        print("-" * 78)

        vr = {"video": tag, "resolution": f"{w}x{h}", "sample_frames": len(frames)}

        # WeChatQRCode baseline
        t = bench_wechat(frames, ITERS)
        s = pr("OpenCV WeChatQRCode (full)", t)
        base = s.get("p50", 1) if s else 1
        vr["wechat"] = s

        # MNN single-frame
        for be in ["CPU", "METAL"]:
            t = bench_mnn_single(frames, ITERS, be)
            s = pr(f"MNN single ({be})", t, base)
            vr[f"mnn_single_{be.lower()}"] = s

        # MNN batch
        for bs in BATCH_SIZES:
            for be in ["CPU", "METAL"]:
                t = bench_mnn_batch(frames, bs, ITERS, be)
                s = pr(f"MNN batch={bs:2d} ({be})", t, base)
                vr[f"mnn_batch{bs}_{be.lower()}"] = s

        all_results[tag] = vr
        print()

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "bench_real_frames.json"
    report = {
        "platform": platform.platform(),
        "mnn": MNN.version(),
        "opencv": cv2.__version__,
        "iters": ITERS,
        "batch_sizes": BATCH_SIZES,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
