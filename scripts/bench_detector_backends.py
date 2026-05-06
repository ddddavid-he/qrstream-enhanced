#!/usr/bin/env python3
"""Benchmark wechat-qrcode vs zxing-cpp on real QRStream video captures.

Methodology:
1. Run the probe phase on each video to learn crop_box and adaptive_max_dim.
2. Extract every frame that would enter the main scan pipeline
   (apply crop + downscale exactly as the decoder does).
3. Run both detectors on each prepared frame slice and record:
   - Detection success (decoded a non-empty string)
   - Wall-clock time per frame

Videos used:
  - IMG_9442.MOV  (~10 s, smallest)
  - IMG_9455.MOV  (~18 s)
  - IMG_9448.MOV  (only first 20 s extracted via ffmpeg)

Usage:
  uv run python scripts/bench_detector_backends.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, median, stdev

import cv2
import numpy as np

# ── project imports ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qrstream.decoder import (
    _MAX_DETECT_DIM,
    _PROBE_MAX_DETECT_DIM,
    _crop_frame,
    _downscale_frame,
    _probe_sample_rate,
)
from qrstream.qr_utils import try_decode_qr

import zxingcpp

# ── configuration ─────────────────────────────────────────────────
TESTCASE_DIR = Path("/Users/ddddavid/Downloads/testcase")
VIDEOS = [
    TESTCASE_DIR / "IMG_9442.MOV",
    TESTCASE_DIR / "IMG_9455.MOV",
    TESTCASE_DIR / "IMG_9448.MOV",  # will be clipped to 20 s
]
CLIP_DURATION = 20  # seconds, only applied to videos > this duration


# ── zxing-cpp detector ────────────────────────────────────────────

def decode_zxing(frame: np.ndarray) -> str | None:
    """Decode a QR code from a BGR frame using zxing-cpp."""
    result = zxingcpp.read_barcode(
        frame,
        formats=zxingcpp.QRCode,
        try_rotate=True,
        try_downscale=False,  # we already handle downscaling ourselves
        try_invert=False,
    )
    if result is None or not result.valid:
        return None
    return result.text or None


# ── wechat detector (re-use thread-local singleton) ───────────────

def decode_wechat(frame: np.ndarray) -> str | None:
    """Decode a QR code from a BGR frame using WeChatQRCode."""
    return try_decode_qr(frame)


# ── helpers ───────────────────────────────────────────────────────

def clip_video(src: Path, duration: int) -> Path:
    """Return a path to a 20-s clip of *src* (temp file, caller owns it)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-t", str(duration),
        "-c:v", "copy", "-an",
        tmp.name,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return Path(tmp.name)


def get_video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps


def prepare_video(path: Path) -> tuple[Path, bool]:
    """Return (working_path, is_temp). Clips long videos to CLIP_DURATION s."""
    dur = get_video_duration(path)
    if dur > CLIP_DURATION + 2:
        print(f"  Clipping {path.name} ({dur:.0f}s) → {CLIP_DURATION}s …")
        return clip_video(path, CLIP_DURATION), True
    return path, False


def run_probe(video_path: Path) -> tuple[int, tuple | None, int, float]:
    """Run decoder probe phase; return (sample_rate, crop_box, adaptive_max_dim, detect_rate)."""
    print("  Running probe phase …")
    t0 = time.perf_counter()
    (
        sample_rate,
        _probe_results,
        _probe_count,
        _leading,
        detect_rate,
        _avg_repeat,
        adaptive_max_dim,
        crop_box,
    ) = _probe_sample_rate(str(video_path), workers=4, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"  Probe done in {elapsed:.1f}s: "
          f"sample_rate={sample_rate}, "
          f"adaptive_max_dim={adaptive_max_dim}, "
          f"crop_box={crop_box}, "
          f"detect_rate={detect_rate:.3f}")
    return sample_rate, crop_box, adaptive_max_dim or _MAX_DETECT_DIM, detect_rate


def extract_frames(
    video_path: Path,
    sample_rate: int,
    crop_box: tuple | None,
    max_dim: int,
) -> list[np.ndarray]:
    """Extract every sample_rate-th frame and apply crop+downscale."""
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            f = _crop_frame(frame, crop_box)
            f = _downscale_frame(f, max_dim=max_dim)
            f = np.ascontiguousarray(f).copy()
            frames.append(f)
        idx += 1
    cap.release()
    return frames


def bench_detector(
    name: str,
    detect_fn,
    frames: list[np.ndarray],
) -> dict:
    """Run detect_fn over all frames; return timing and accuracy stats."""
    times: list[float] = []
    hits = 0
    for f in frames:
        t0 = time.perf_counter()
        result = detect_fn(f)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if result:
            hits += 1

    n = len(frames)
    return {
        "name": name,
        "n_frames": n,
        "hits": hits,
        "hit_rate": hits / n if n else 0.0,
        "mean_ms": mean(times) * 1000 if times else 0.0,
        "median_ms": median(times) * 1000 if times else 0.0,
        "stdev_ms": stdev(times) * 1000 if len(times) > 1 else 0.0,
        "total_s": sum(times),
    }


def print_comparison(video_name: str, wechat: dict, zxing: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {video_name}")
    print(f"{'='*60}")
    print(f"  {'Metric':<22} {'WeChatQR':>12} {'zxing-cpp':>12}")
    print(f"  {'-'*46}")
    for key, label in [
        ("n_frames",  "Frames tested"),
        ("hits",      "QR hits"),
        ("hit_rate",  "Hit rate"),
        ("mean_ms",   "Mean ms/frame"),
        ("median_ms", "Median ms/frame"),
        ("stdev_ms",  "Stdev ms/frame"),
        ("total_s",   "Total time (s)"),
    ]:
        wv = wechat[key]
        zv = zxing[key]
        if isinstance(wv, float):
            print(f"  {label:<22} {wv:>12.3f} {zv:>12.3f}")
        else:
            print(f"  {label:<22} {wv:>12} {zv:>12}")

    # delta summary
    if wechat["hit_rate"] > 0:
        rel = (zxing["hit_rate"] - wechat["hit_rate"]) / wechat["hit_rate"] * 100
    else:
        rel = float("nan")
    speedup = wechat["mean_ms"] / zxing["mean_ms"] if zxing["mean_ms"] > 0 else float("nan")
    print(f"\n  Hit-rate delta   : {zxing['hit_rate']-wechat['hit_rate']:+.3f} ({rel:+.1f}%)")
    print(f"  Speed ratio      : zxing is {speedup:.2f}× vs wechat (mean)")


# ── main ──────────────────────────────────────────────────────────

def main() -> None:
    all_results: list[tuple[str, dict, dict]] = []

    for src in VIDEOS:
        if not src.exists():
            print(f"SKIP (not found): {src}")
            continue

        print(f"\n{'#'*60}")
        print(f"# {src.name}")
        print(f"{'#'*60}")

        video_path, is_temp = prepare_video(src)
        try:
            sample_rate, crop_box, max_dim, detect_rate = run_probe(video_path)

            print(f"  Extracting frames (every {sample_rate}th, "
                  f"max_dim={max_dim}, crop={crop_box is not None}) …")
            frames = extract_frames(video_path, sample_rate, crop_box, max_dim)
            print(f"  → {len(frames)} frames extracted")

            if not frames:
                print("  No frames — skipping.")
                continue

            print("  Benchmarking WeChatQR …")
            wechat_res = bench_detector("wechat", decode_wechat, frames)

            print("  Benchmarking zxing-cpp …")
            zxing_res = bench_detector("zxing", decode_zxing, frames)

            all_results.append((src.name, wechat_res, zxing_res))
            print_comparison(src.name, wechat_res, zxing_res)

        finally:
            if is_temp:
                video_path.unlink(missing_ok=True)

    # ── aggregate summary ─────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  AGGREGATE SUMMARY (all videos)")
        print(f"{'='*60}")
        total_frames = sum(r[1]["n_frames"] for r in all_results)
        wechat_hits = sum(r[1]["hits"] for r in all_results)
        zxing_hits  = sum(r[2]["hits"] for r in all_results)
        wechat_total_s = sum(r[1]["total_s"] for r in all_results)
        zxing_total_s  = sum(r[2]["total_s"] for r in all_results)

        print(f"  Total frames     : {total_frames}")
        print(f"  WeChatQR hits    : {wechat_hits} ({wechat_hits/total_frames:.3f})")
        print(f"  zxing-cpp hits   : {zxing_hits} ({zxing_hits/total_frames:.3f})")
        print(f"  WeChatQR total s : {wechat_total_s:.2f}")
        print(f"  zxing-cpp total s: {zxing_total_s:.2f}")
        speedup = wechat_total_s / zxing_total_s if zxing_total_s > 0 else float("nan")
        delta_hr = zxing_hits / total_frames - wechat_hits / total_frames
        print(f"  Hit-rate delta   : {delta_hr:+.3f}")
        print(f"  Speed ratio      : zxing is {speedup:.2f}× vs wechat (total)")


if __name__ == "__main__":
    main()
