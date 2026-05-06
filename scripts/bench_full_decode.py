#!/usr/bin/env python3
"""Full-decode benchmark: wechat-qrcode vs zxing-cpp on long videos.

Runs a complete extract_qr_from_video → decode_blocks pipeline with each
detector backend and measures:
  - Wall-clock time (probe + scan + recovery)
  - Unique blocks recovered
  - Whether the file reconstructed correctly (SHA-256 vs reference .bin)
  - LT-decoder convergence / recovery behaviour

Backend swap strategy:
  The decoder module exposes a module-level ``_dispatch_detect`` hook that
  ``_worker_detect_qr`` calls instead of ``try_decode_qr`` directly.
  We monkey-patch this hook (and its bbox-returning twin used by the probe)
  to route calls through zxing-cpp.  The sandbox is disabled for both runs
  (detect_isolation='off') so neither backend gets a subprocess overhead
  advantage — this isolates the raw detector speed difference.

Usage:
  uv run python scripts/bench_full_decode.py [--video 9448|9432|both]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import qrstream.decoder as _dec_mod
from qrstream.decoder import (
    decode_blocks,
    extract_qr_from_video,
)
from qrstream.qr_utils import try_decode_qr, try_decode_qr_with_bbox

# ── zxing-cpp detector shims ──────────────────────────────────────

def _zxing_detect(frame_idx: int, frame: np.ndarray) -> str | None:  # noqa: ARG001
    result = zxingcpp.read_barcode(
        frame,
        formats=zxingcpp.QRCode,
        try_rotate=True,
        try_downscale=False,
        try_invert=False,
    )
    if result is None or not result.valid:
        return None
    return result.text or None


def _zxing_detect_with_bbox(
    frame_idx: int, frame: np.ndarray  # noqa: ARG001
) -> tuple | None:
    """Return (text, bbox_ndarray) like try_decode_qr_with_bbox, or None."""
    result = zxingcpp.read_barcode(
        frame,
        formats=zxingcpp.QRCode,
        try_rotate=True,
        try_downscale=False,
        try_invert=False,
    )
    if result is None or not result.valid or not result.text:
        return None
    # Build a (4,2) float32 bbox from zxing position if available.
    pos = result.position
    try:
        bbox = np.array([
            [pos.top_left.x,     pos.top_left.y],
            [pos.top_right.x,    pos.top_right.y],
            [pos.bottom_right.x, pos.bottom_right.y],
            [pos.bottom_left.x,  pos.bottom_left.y],
        ], dtype=np.float32)
    except Exception:
        bbox = np.zeros((4, 2), dtype=np.float32)
    return (result.text, bbox)


# ── wechat-qrcode shims (matching the same signature) ────────────

def _wechat_detect(frame_idx: int, frame: np.ndarray) -> str | None:  # noqa: ARG001
    return try_decode_qr(frame)


def _wechat_detect_with_bbox(
    frame_idx: int, frame: np.ndarray  # noqa: ARG001
) -> tuple | None:
    return try_decode_qr_with_bbox(frame)


# ── helpers ───────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_full_decode(
    video_path: Path,
    backend_name: str,
    detect_fn,
    detect_bbox_fn,
    ref_path: Path | None,
    out_path: Path,
) -> dict:
    """Run extract + decode with the given backend; return result dict."""
    # Patch dispatch hooks
    _dec_mod._dispatch_detect = detect_fn
    _dec_mod._dispatch_detect_with_bbox = detect_bbox_fn

    t0 = time.perf_counter()
    blocks = extract_qr_from_video(
        str(video_path),
        sample_rate=0,       # auto-probe
        verbose=False,
        workers=4,
        detect_isolation="off",  # bypass sandbox for fair comparison
    )
    t_extract = time.perf_counter() - t0

    n_blocks = len(blocks) if blocks else 0

    t1 = time.perf_counter()
    data = decode_blocks(blocks) if blocks else None
    t_decode = time.perf_counter() - t1

    total = time.perf_counter() - t0

    # Write decoded output
    ok = False
    hash_match: bool | None = None
    decoded_size = 0

    if data is not None:
        decoded_size = len(data)
        out_path.write_bytes(data)
        ok = True
        if ref_path and ref_path.exists():
            ref_hash = sha256_file(ref_path)
            dec_hash = sha256_bytes(data)
            hash_match = ref_hash == dec_hash

    return {
        "backend": backend_name,
        "video": video_path.name,
        "blocks_recovered": n_blocks,
        "decode_success": ok,
        "decoded_size_bytes": decoded_size,
        "hash_match": hash_match,
        "t_extract_s": t_extract,
        "t_decode_s": t_decode,
        "total_s": total,
    }


def print_result(r: dict) -> None:
    ok_str   = "YES" if r["decode_success"] else "NO"
    hash_str = ("MATCH" if r["hash_match"] else "MISMATCH") if r["hash_match"] is not None else "N/A"
    print(f"  Backend          : {r['backend']}")
    print(f"  Blocks recovered : {r['blocks_recovered']}")
    print(f"  Decode success   : {ok_str}")
    print(f"  Decoded size     : {r['decoded_size_bytes']:,} bytes")
    print(f"  Hash vs ref      : {hash_str}")
    print(f"  Extract time     : {r['t_extract_s']:.1f} s")
    print(f"  LT decode time   : {r['t_decode_s']:.3f} s")
    print(f"  Total time       : {r['total_s']:.1f} s")


def compare_pair(wechat: dict, zxing: dict) -> None:
    print(f"\n  {'Metric':<24} {'WeChatQR':>12} {'zxing-cpp':>12}")
    print(f"  {'-'*48}")
    rows = [
        ("Blocks recovered",  "blocks_recovered"),
        ("Decode success",    "decode_success"),
        ("Decoded size (B)",  "decoded_size_bytes"),
        ("Hash match",        "hash_match"),
        ("Extract time (s)",  "t_extract_s"),
        ("Total time (s)",    "total_s"),
    ]
    for label, key in rows:
        wv = wechat[key]
        zv = zxing[key]
        if isinstance(wv, float):
            print(f"  {label:<24} {wv:>12.1f} {zv:>12.1f}")
        else:
            print(f"  {label:<24} {str(wv):>12} {str(zv):>12}")

    speedup = wechat["total_s"] / zxing["total_s"] if zxing["total_s"] > 0 else float("nan")
    print(f"\n  Speed ratio      : zxing is {speedup:.2f}× faster (total)")


# ── main ──────────────────────────────────────────────────────────

TESTCASE = Path("/Users/ddddavid/Downloads/testcase")

VIDEO_CONFIGS = {
    "9432": {
        "video": TESTCASE / "IMG_9432.MOV",
        "ref":   TESTCASE / "9432.bin",
    },
    "9448": {
        "video": TESTCASE / "IMG_9448.MOV",
        "ref":   TESTCASE / "9448.bin",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="both",
                        choices=["9432", "9448", "both"],
                        help="Which video(s) to test (default: both)")
    parser.add_argument("--backend", default="both",
                        choices=["wechat", "zxing", "both"],
                        help="Which backend(s) to test (default: both)")
    args = parser.parse_args()

    targets = list(VIDEO_CONFIGS.keys()) if args.video == "both" else [args.video]

    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="bench_decode_"))

    for vid_key in targets:
        cfg = VIDEO_CONFIGS[vid_key]
        video_path: Path = cfg["video"]
        ref_path: Path   = cfg["ref"]

        if not video_path.exists():
            print(f"SKIP (not found): {video_path}")
            continue

        dur = int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_COUNT) /
                  (cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS) or 30))
        print(f"\n{'#'*62}")
        print(f"# {video_path.name}  ({dur}s)")
        print(f"{'#'*62}")

        r_wechat = None
        r_zxing = None

        # ── WeChatQR ─────────────────────────────────────────────
        if args.backend in ("wechat", "both"):
            print("\n[wechat] WeChatQR backend …")
            wechat_out = tmpdir / f"{vid_key}_wechat.bin"
            r_wechat = run_full_decode(
                video_path, "wechat",
                _wechat_detect, _wechat_detect_with_bbox,
                ref_path, wechat_out,
            )
            print_result(r_wechat)

        # ── zxing-cpp ────────────────────────────────────────────
        if args.backend in ("zxing", "both"):
            print("\n[zxing] zxing-cpp backend …")
            zxing_out = tmpdir / f"{vid_key}_zxing.bin"
            r_zxing = run_full_decode(
                video_path, "zxing",
                _zxing_detect, _zxing_detect_with_bbox,
                ref_path, zxing_out,
            )
            print_result(r_zxing)

        # ── side-by-side ─────────────────────────────────────────
        if r_wechat is not None and r_zxing is not None:
            print(f"\n{'='*62}")
            print(f"  Side-by-side: {video_path.name}")
            print(f"{'='*62}")
            compare_pair(r_wechat, r_zxing)

    print(f"\nOutput files in: {tmpdir}")


if __name__ == "__main__":
    main()
