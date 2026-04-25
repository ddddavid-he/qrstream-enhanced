#!/usr/bin/env python3
"""
Single-frame latency benchmark: OpenCV DNN vs MNN.

Measures per-frame QR detection latency across multiple backends and
reports P50 / P95 / P99 / mean statistics.

Usage:
    python benchmark_single_frame.py --image test_qr.png --iterations 100
    python benchmark_single_frame.py --image test_qr.png --backend metal
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def benchmark_opencv_wechat(img: np.ndarray, iterations: int) -> list[float]:
    """Benchmark OpenCV WeChatQRCode detectAndDecode."""
    detector = cv2.wechat_qrcode_WeChatQRCode()

    # Warmup
    for _ in range(3):
        detector.detectAndDecode(img)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        results, _ = detector.detectAndDecode(img)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return times


def benchmark_mnn_detector(img: np.ndarray, iterations: int,
                           backend: str = "cpu") -> list[float]:
    """Benchmark MNN detector (detect + SR + decode)."""
    try:
        from qrstream.detector import DetectorRouter
    except ImportError:
        print("ERROR: qrstream.detector not importable")
        return []

    router = DetectorRouter(use_mnn=True, mnn_backend=backend)

    # Warmup
    for _ in range(3):
        router.detect(img)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = router.detect(img)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return times


def print_stats(label: str, times: list[float]):
    """Print percentile statistics."""
    if not times:
        print(f"  {label}: no data")
        return

    arr = np.array(times)
    print(f"  {label}:")
    print(f"    P50:  {np.percentile(arr, 50):.2f} ms")
    print(f"    P95:  {np.percentile(arr, 95):.2f} ms")
    print(f"    P99:  {np.percentile(arr, 99):.2f} ms")
    print(f"    Mean: {np.mean(arr):.2f} ms")
    print(f"    Std:  {np.std(arr):.2f} ms")
    print(f"    Min:  {np.min(arr):.2f} ms")
    print(f"    Max:  {np.max(arr):.2f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Single-frame QR detection latency benchmark")
    parser.add_argument("--image", required=True, help="Test image path")
    parser.add_argument("--iterations", "-n", type=int, default=100,
                        help="Number of iterations (default: 100)")
    parser.add_argument("--backend", default="cpu",
                        choices=["cpu", "metal", "cuda", "opencl"],
                        help="MNN backend to benchmark (default: cpu)")
    parser.add_argument("--output", "-o", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Cannot read image: {args.image}")
        sys.exit(1)

    print(f"Image: {args.image} ({img.shape[1]}x{img.shape[0]})")
    print(f"Iterations: {args.iterations}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    print()

    results = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "python": sys.version,
        "image": args.image,
        "iterations": args.iterations,
    }

    # OpenCV baseline
    print("Benchmarking OpenCV WeChatQRCode...")
    opencv_times = benchmark_opencv_wechat(img, args.iterations)
    print_stats("OpenCV WeChatQRCode", opencv_times)
    results["opencv_wechat"] = {
        "p50": float(np.percentile(opencv_times, 50)),
        "p95": float(np.percentile(opencv_times, 95)),
        "mean": float(np.mean(opencv_times)),
    }

    # MNN
    print(f"\nBenchmarking MNN ({args.backend})...")
    mnn_times = benchmark_mnn_detector(img, args.iterations, args.backend)
    if mnn_times:
        print_stats(f"MNN ({args.backend})", mnn_times)
        results[f"mnn_{args.backend}"] = {
            "p50": float(np.percentile(mnn_times, 50)),
            "p95": float(np.percentile(mnn_times, 95)),
            "mean": float(np.mean(mnn_times)),
        }

        # Speedup
        speedup = np.mean(opencv_times) / np.mean(mnn_times)
        print(f"\n  Speedup (P50): {np.percentile(opencv_times, 50) / np.percentile(mnn_times, 50):.2f}x")
        print(f"  Speedup (mean): {speedup:.2f}x")
        results["speedup_mean"] = float(speedup)
    else:
        print("  MNN: skipped (not available)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
