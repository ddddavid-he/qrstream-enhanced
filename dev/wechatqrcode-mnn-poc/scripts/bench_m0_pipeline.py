#!/usr/bin/env python3
"""
Milestone 0 Pipeline Benchmark: OpenCV DNN vs MNN (CPU / Metal)

Simulates the actual qrstream decode pipeline's per-frame QR detection
path, measuring single-frame latency for each backend.

Test cases:
  1. Easy: clean QR on white background (400×400)
  2. Medium: QR with noise and rotation (640×480)
  3. Hard: small QR in large noisy frame (1080×720)
  4. Worst: empty frame with no QR (measures no-detect overhead)

For each test case we measure:
  - OpenCV WeChatQRCode (current production path)
  - OpenCV DNN detector only (Caffe, CPU)
  - MNN detector only (CPU)
  - MNN detector only (Metal, if available)

Reports P50 / P95 / mean latency and speedup ratios.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "models"
CAFFE_DIR = MODEL_DIR / "caffe"
MNN_DIR = MODEL_DIR / "mnn"

SSD_TARGET_AREA = 400.0 * 400.0
WARMUP = 5
ITERATIONS = 100


# ── Test image generation ────────────────────────────────────────

def make_qr_image(text: str, size: int = 400) -> np.ndarray:
    """Generate a clean QR on white canvas."""
    try:
        enc = cv2.QRCodeEncoder.create()
        qr = enc.encode(text)
        if qr is not None and qr.size > 0:
            if qr.ndim == 2:
                qr = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
            h, w = qr.shape[:2]
            canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
            y0 = (size - h) // 2
            x0 = (size - w) // 2
            y1 = min(y0 + h, size)
            x1 = min(x0 + w, size)
            canvas[y0:y1, x0:x1] = qr[:y1 - y0, :x1 - x0]
            return canvas
    except Exception:
        pass
    # Fallback: synthetic pattern
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 220
    cv2.rectangle(canvas, (size // 4, size // 4),
                  (3 * size // 4, 3 * size // 4), (0, 0, 0), 2)
    return canvas


def make_noisy_qr(size_w: int = 640, size_h: int = 480) -> np.ndarray:
    """QR embedded in a noisy, slightly rotated frame."""
    qr_img = make_qr_image("MEDIUM DIFFICULTY TEST PAYLOAD 12345", 200)
    canvas = np.random.randint(180, 240, (size_h, size_w, 3), dtype=np.uint8)
    # Rotate QR slightly
    M = cv2.getRotationMatrix2D((100, 100), 5, 1.0)
    qr_rot = cv2.warpAffine(qr_img, M, (200, 200),
                             borderValue=(220, 220, 220))
    y0 = (size_h - 200) // 2
    x0 = (size_w - 200) // 2
    canvas[y0:y0 + 200, x0:x0 + 200] = qr_rot
    # Add Gaussian noise
    noise = np.random.normal(0, 15, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def make_hard_frame(size_w: int = 1080, size_h: int = 720) -> np.ndarray:
    """Small QR in a large noisy frame."""
    qr_img = make_qr_image("HARD CASE SMALL QR", 120)
    canvas = np.random.randint(100, 200, (size_h, size_w, 3), dtype=np.uint8)
    # Place QR in bottom-right area
    y0 = size_h - 150
    x0 = size_w - 150
    canvas[y0:y0 + 120, x0:x0 + 120] = qr_img[:120, :120]
    noise = np.random.normal(0, 20, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def make_empty_frame(size_w: int = 640, size_h: int = 480) -> np.ndarray:
    """Frame with no QR — measures no-detect overhead."""
    canvas = np.random.randint(100, 200, (size_h, size_w, 3), dtype=np.uint8)
    noise = np.random.normal(0, 25, canvas.shape).astype(np.int16)
    return np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ── Preprocessing helpers ────────────────────────────────────────

def preprocess_for_ssd(frame: np.ndarray):
    """Convert frame to grayscale, resize for SSD, normalize."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = min(1.0, (SSD_TARGET_AREA / (w * h)) ** 0.5)
    det_w = int(w * scale)
    det_h = int(h * scale)
    resized = cv2.resize(gray, (det_w, det_h), interpolation=cv2.INTER_CUBIC)
    return resized, det_w, det_h, w, h


# ── Benchmark runners ────────────────────────────────────────────

def bench_opencv_wechat(frame: np.ndarray, n: int) -> list[float]:
    """Full OpenCV WeChatQRCode detectAndDecode (production path)."""
    det = cv2.wechat_qrcode_WeChatQRCode()
    for _ in range(WARMUP):
        det.detectAndDecode(frame)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        det.detectAndDecode(frame)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_opencv_dnn_detect(frame: np.ndarray, n: int) -> list[float]:
    """OpenCV DNN Caffe detector only (no decode)."""
    proto = str(CAFFE_DIR / "detect.prototxt")
    model = str(CAFFE_DIR / "detect.caffemodel")
    if not Path(proto).exists():
        return []
    net = cv2.dnn.readNetFromCaffe(proto, model)
    resized, det_w, det_h, img_w, img_h = preprocess_for_ssd(frame)
    blob = cv2.dnn.blobFromImage(resized, 1.0 / 255,
                                  (det_w, det_h), (0.0,), False, False)
    for _ in range(WARMUP):
        net.setInput(blob, "data")
        net.forward("detection_output")
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        net.setInput(blob, "data")
        net.forward("detection_output")
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_mnn_detect(frame: np.ndarray, n: int, backend: str = "CPU") -> list[float]:
    """MNN detector only."""
    import MNN
    mnn_path = str(MNN_DIR / "detect.mnn")
    if not Path(mnn_path).exists():
        return []
    interp = MNN.Interpreter(mnn_path)
    session = interp.createSession({"backend": backend})
    resized, det_w, det_h, img_w, img_h = preprocess_for_ssd(frame)
    input_data = resized.astype(np.float32) / 255.0
    input_data = input_data.reshape(1, 1, det_h, det_w)

    inp = interp.getSessionInput(session)
    interp.resizeTensor(inp, (1, 1, det_h, det_w))
    interp.resizeSession(session)

    def run_once():
        tmp_in = MNN.Tensor((1, 1, det_h, det_w), MNN.Halide_Type_Float,
                            input_data, MNN.Tensor_DimensionType_Caffe)
        inp.copyFrom(tmp_in)
        interp.runSession(session)
        out = interp.getSessionOutput(session, "detection_output")
        out_shape = out.getShape()
        if out_shape[2] > 0:
            tmp_out = MNN.Tensor(out_shape, MNN.Halide_Type_Float,
                                 np.zeros(out_shape, dtype=np.float32),
                                 MNN.Tensor_DimensionType_Caffe)
            out.copyToHostTensor(tmp_out)

    for _ in range(WARMUP):
        run_once()

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        run_once()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_mnn_full_pipeline(frame: np.ndarray, n: int, backend: str = "CPU") -> list[float]:
    """MNN detector + SR + CPU decode (simulated full pipeline)."""
    import MNN
    det_path = str(MNN_DIR / "detect.mnn")
    sr_path = str(MNN_DIR / "sr.mnn")
    if not Path(det_path).exists():
        return []

    # Setup detector
    det_interp = MNN.Interpreter(det_path)
    det_session = det_interp.createSession({"backend": backend})
    resized, det_w, det_h, img_w, img_h = preprocess_for_ssd(frame)
    det_inp = det_interp.getSessionInput(det_session)
    det_interp.resizeTensor(det_inp, (1, 1, det_h, det_w))
    det_interp.resizeSession(det_session)
    det_input = (resized.astype(np.float32) / 255.0).reshape(1, 1, det_h, det_w)

    # Setup SR
    has_sr = Path(sr_path).exists()
    sr_interp = sr_session = None
    if has_sr:
        sr_interp = MNN.Interpreter(sr_path)
        sr_session = sr_interp.createSession({"backend": backend})

    # WeChatQRCode for CPU decode
    wechat = cv2.wechat_qrcode_WeChatQRCode()

    def run_once():
        # 1. Detect
        tmp_in = MNN.Tensor((1, 1, det_h, det_w), MNN.Halide_Type_Float,
                            det_input, MNN.Tensor_DimensionType_Caffe)
        det_inp.copyFrom(tmp_in)
        det_interp.runSession(det_session)
        out = det_interp.getSessionOutput(det_session, "detection_output")
        out_shape = out.getShape()
        if out_shape[2] == 0:
            return  # no detection

        tmp_out = MNN.Tensor(out_shape, MNN.Halide_Type_Float,
                             np.zeros(out_shape, dtype=np.float32),
                             MNN.Tensor_DimensionType_Caffe)
        out.copyToHostTensor(tmp_out)
        data = np.array(tmp_out.getData(), dtype=np.float32)
        n_det = out_shape[2]
        dim = out_shape[3]
        if len(data) < n_det * dim:
            return
        arr = data[:n_det * dim].reshape(n_det, dim)

        for row in arr:
            if dim == 6:
                cls, conf = row[0], row[1]
                x0, y0, x1, y1 = row[2], row[3], row[4], row[5]
            else:
                cls, conf = row[1], row[2]
                x0, y0, x1, y1 = row[3], row[4], row[5], row[6]
            if cls != 1 or conf <= 1e-5:
                continue

            # 2. Crop ROI
            px0 = max(0, int(x0 * img_w))
            py0 = max(0, int(y0 * img_h))
            px1 = min(img_w, int(np.ceil(x1 * img_w)))
            py1 = min(img_h, int(np.ceil(y1 * img_h)))
            if px1 <= px0 or py1 <= py0:
                continue
            crop = frame[py0:py1, px0:px1]

            # 3. Optional SR
            if has_sr and sr_interp is not None:
                area = (px1 - px0) * (py1 - py0)
                if area > 0 and int(area ** 0.5) < 160:
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    ch, cw = gray_crop.shape
                    sr_inp_t = sr_interp.getSessionInput(sr_session)
                    sr_interp.resizeTensor(sr_inp_t, (1, 1, ch, cw))
                    sr_interp.resizeSession(sr_session)
                    sr_data = (gray_crop.astype(np.float32) / 255.0).reshape(1, 1, ch, cw)
                    tmp_sr = MNN.Tensor((1, 1, ch, cw), MNN.Halide_Type_Float,
                                        sr_data, MNN.Tensor_DimensionType_Caffe)
                    sr_inp_t.copyFrom(tmp_sr)
                    sr_interp.runSession(sr_session)

            # 4. CPU decode
            wechat.detectAndDecode(crop)
            break  # first detection only

    for _ in range(WARMUP):
        run_once()

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        run_once()
        times.append((time.perf_counter() - t0) * 1000)
    return times


# ── Reporting ────────────────────────────────────────────────────

def stats(times: list[float]) -> dict:
    if not times:
        return {"p50": 0, "p95": 0, "mean": 0, "min": 0, "max": 0}
    a = np.array(times)
    return {
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "mean": round(float(np.mean(a)), 3),
        "min": round(float(np.min(a)), 3),
        "max": round(float(np.max(a)), 3),
    }


def print_row(label: str, times: list[float], baseline_p50: float = 0):
    s = stats(times)
    if not times:
        print(f"  {label:35s}  (skipped)")
        return s
    speedup = ""
    if baseline_p50 > 0 and s["p50"] > 0:
        speedup = f"  {baseline_p50 / s['p50']:.2f}x"
    print(f"  {label:35s}  P50={s['p50']:7.2f}ms  P95={s['p95']:7.2f}ms  "
          f"mean={s['mean']:7.2f}ms  min={s['min']:7.2f}ms{speedup}")
    return s


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("QRStream MNN PoC — Pipeline Benchmark")
    print("=" * 75)
    print(f"Platform:    {platform.platform()}")
    print(f"CPU:         {platform.processor() or platform.machine()}")
    print(f"Python:      {sys.version.split()[0]}")
    print(f"OpenCV:      {cv2.__version__}")
    try:
        import MNN
        print(f"MNN:         {MNN.version()}")
    except ImportError:
        print("MNN:         NOT INSTALLED")
        return
    print(f"Iterations:  {ITERATIONS} (warmup: {WARMUP})")
    print()

    test_cases = [
        ("Easy: clean QR 400×400",     make_qr_image("QRSTREAM BENCH EASY", 400)),
        ("Medium: noisy QR 640×480",   make_noisy_qr(640, 480)),
        ("Hard: small QR 1080×720",    make_hard_frame(1080, 720)),
        ("No QR: empty 640×480",       make_empty_frame(640, 480)),
    ]

    all_results = {}

    for case_name, frame in test_cases:
        print("-" * 75)
        print(f"Test: {case_name}  (frame: {frame.shape[1]}×{frame.shape[0]})")
        print("-" * 75)

        case_data = {"frame_size": f"{frame.shape[1]}x{frame.shape[0]}"}

        # OpenCV WeChatQRCode (production baseline)
        t = bench_opencv_wechat(frame, ITERATIONS)
        s = print_row("OpenCV WeChatQRCode (full)", t)
        baseline_full = s["p50"]
        case_data["opencv_wechat_full"] = s

        # OpenCV DNN detector only
        t = bench_opencv_dnn_detect(frame, ITERATIONS)
        s = print_row("OpenCV DNN detect only", t, baseline_full)
        case_data["opencv_dnn_detect"] = s

        # MNN CPU detect only
        t = bench_mnn_detect(frame, ITERATIONS, "CPU")
        s = print_row("MNN detect only (CPU)", t, baseline_full)
        case_data["mnn_cpu_detect"] = s

        # MNN Metal detect only
        t = bench_mnn_detect(frame, ITERATIONS, "METAL")
        s = print_row("MNN detect only (Metal)", t, baseline_full)
        case_data["mnn_metal_detect"] = s

        # MNN CPU full pipeline (detect + SR + decode)
        t = bench_mnn_full_pipeline(frame, ITERATIONS, "CPU")
        s = print_row("MNN full pipeline (CPU)", t, baseline_full)
        case_data["mnn_cpu_pipeline"] = s

        # MNN Metal full pipeline
        t = bench_mnn_full_pipeline(frame, ITERATIONS, "METAL")
        s = print_row("MNN full pipeline (Metal)", t, baseline_full)
        case_data["mnn_metal_pipeline"] = s

        all_results[case_name] = case_data
        print()

    # Save JSON
    report_path = SCRIPT_DIR.parent / "results" / "bench_m0_pipeline.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    full_report = {
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "mnn": MNN.version(),
        "iterations": ITERATIONS,
        "results": all_results,
    }
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
