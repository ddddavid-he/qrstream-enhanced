#!/usr/bin/env python3
"""
Milestone 1.75 — CPU decoder probe / survey.

Evaluates candidate QR decoders on cropped regions produced by
MNN SSD + quiet-zone padding (the actual M1.5 production path).

For each video, the script:
  1. Extracts up to --sample-count frames evenly spaced.
  2. Runs MNN CPU detector to get padded crops.
  3. Feeds each crop to every candidate decoder.
  4. Records hit_rate, avg_ms, p95_ms, and the set of frames where
     only WeChatQRCode (the current _cpu_decode backend) succeeds
     but the candidate fails ("exclusive WeChatQR misses").

Candidates:
  A. cv2.wechat_qrcode_WeChatQRCode  (current _cpu_decode — baseline)
  B. cv2.QRCodeDetector               (lightweight OpenCV native)
  C. zxing-cpp (zxingcpp)             (if installed)

Output: JSON + Markdown summary to stdout and optionally to a file.

Usage inside Containerfile.m175 (or locally with MNN + models):

    python probe_cpu_decoders.py --videos tests/fixtures/ \\
        --sample-count 200 --output results/cpu_decoder_survey.md
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── MNN import (will fail outside container unless MNN installed) ──
try:
    import MNN
    _HAS_MNN = True
except ImportError:
    _HAS_MNN = False

# ── zxing-cpp import (candidate C) ─────────────────────────────────
try:
    import zxingcpp
    _HAS_ZXING_CPP = True
except ImportError:
    _HAS_ZXING_CPP = False


SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent.parent.parent
MNN_MODEL_DIR = SCRIPT_DIR.parent / "models" / "mnn"

# Parameters matching MNNQrDetector production path
_SSD_TARGET_AREA = 400.0 * 400.0
_QUIET_ZONE_PAD_RATIO = 0.15
_SR_MAX_SIZE = 160
_MAX_DET_DIM = 1080


# ── Frame extraction ──────────────────────────────────────────────

def extract_frames(video_path: str, count: int = 200) -> list[np.ndarray]:
    """Extract up to ``count`` evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, min(count, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            mx = max(h, w)
            if mx > _MAX_DET_DIM:
                s = _MAX_DET_DIM / mx
                frame = cv2.resize(
                    frame, (int(w * s), int(h * s)),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(frame)
    cap.release()
    return frames


# ── MNN detector (replicates MNNQrDetector._run_detector logic) ──

class _MNNDetector:
    """Minimal MNN SSD detector for crop extraction."""

    def __init__(self, model_path: str, backend: str = "CPU"):
        self._interp = MNN.Interpreter(model_path)
        self._session = self._interp.createSession({"backend": backend})
        self._inp = self._interp.getSessionInput(self._session)

    def detect_bboxes(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (x0, y0, x1, y1) pixel bboxes."""
        img_h, img_w = frame.shape[:2]
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        scale = min(1.0, (_SSD_TARGET_AREA / (img_w * img_h)) ** 0.5)
        det_w = int(img_w * scale)
        det_h = int(img_h * scale)
        resized = cv2.resize(gray, (det_w, det_h), interpolation=cv2.INTER_CUBIC)
        data = (resized.astype(np.float32) / 255.0).reshape(1, 1, det_h, det_w)

        self._interp.resizeTensor(self._inp, (1, 1, det_h, det_w))
        self._interp.resizeSession(self._session)

        tmp = MNN.Tensor(
            (1, 1, det_h, det_w), MNN.Halide_Type_Float,
            data, MNN.Tensor_DimensionType_Caffe,
        )
        self._inp.copyFrom(tmp)
        self._interp.runSession(self._session)

        out = self._interp.getSessionOutput(self._session, "detection_output")
        shape = out.getShape()
        if len(shape) < 4 or shape[2] == 0:
            return []

        n_det, dim = shape[2], shape[3]
        if dim not in (6, 7):
            return []
        tmp_out = MNN.Tensor(
            shape, MNN.Halide_Type_Float,
            np.zeros(shape, dtype=np.float32),
            MNN.Tensor_DimensionType_Caffe,
        )
        out.copyToHostTensor(tmp_out)
        arr = np.array(tmp_out.getData(), dtype=np.float32)
        if len(arr) < n_det * dim:
            return []
        arr = arr[:n_det * dim].reshape(n_det, dim)

        bboxes = []
        for row in arr:
            if dim == 7:
                cls, conf = row[1], row[2]
                x0n, y0n, x1n, y1n = row[3], row[4], row[5], row[6]
            else:
                cls, conf = row[0], row[1]
                x0n, y0n, x1n, y1n = row[2], row[3], row[4], row[5]
            if cls != 1 or conf <= 1e-5:
                continue
            x0 = max(0, int(x0n * img_w))
            y0 = max(0, int(y0n * img_h))
            x1 = min(img_w, int(np.ceil(x1n * img_w)))
            y1 = min(img_h, int(np.ceil(y1n * img_h)))
            if x1 <= x0 or y1 <= y0:
                continue
            bboxes.append((x0, y0, x1, y1))
        return bboxes


def _pad_bbox(x0, y0, x1, y1, img_w, img_h, ratio=_QUIET_ZONE_PAD_RATIO):
    """Replicate _pad_bbox from mnn_detector.py."""
    if ratio <= 0:
        return (x0, y0, x1, y1)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return (x0, y0, x1, y1)
    short = min(w, h)
    pad = max(1, int(round(short * ratio)))
    nx0 = max(0, x0 - pad)
    ny0 = max(0, y0 - pad)
    nx1 = min(img_w, x1 + pad)
    ny1 = min(img_h, y1 + pad)
    if nx1 <= nx0 or ny1 <= ny0:
        return (x0, y0, x1, y1)
    return (nx0, ny0, nx1, ny1)


# ── Candidate decoders ────────────────────────────────────────────

def decode_wechat_qr(region: np.ndarray) -> str | None:
    """Current _cpu_decode: OpenCV WeChatQRCode (full pipeline)."""
    if region.ndim == 2:
        region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
    if not region.flags["C_CONTIGUOUS"]:
        region = np.ascontiguousarray(region)
    try:
        det = cv2.wechat_qrcode_WeChatQRCode()
        results, _ = det.detectAndDecode(region)
        if results:
            for r in results:
                if r:
                    return r
    except Exception:
        pass
    return None


def decode_opencv_qr(region: np.ndarray) -> str | None:
    """Candidate B: cv2.QRCodeDetector (lightweight, non-contrib)."""
    if region.ndim == 2:
        region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
    if not region.flags["C_CONTIGUOUS"]:
        region = np.ascontiguousarray(region)
    try:
        det = cv2.QRCodeDetector()
        retval, decoded_info, points, straight = det.detectAndDecodeMulti(region)
        if retval and decoded_info:
            for r in decoded_info:
                if r:
                    return r
    except Exception:
        pass
    return None


def decode_zxing_cpp(region: np.ndarray) -> str | None:
    """Candidate C: zxing-cpp (if installed)."""
    if not _HAS_ZXING_CPP:
        return None
    if region.ndim == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region
    try:
        results = zxingcpp.read_barcodes(gray)
        for r in results:
            if r.text:
                return r.text
    except Exception:
        pass
    return None


def decode_zxing_cpp_multi(region: np.ndarray) -> str | None:
    """Candidate D: zxing-cpp with multi-binarization retry.

    Mirrors WeChatQRCode's multi-binarizer + inversion strategy:
      1. LocalAverage (default)
      2. GlobalHistogram
      3. Gaussian adaptive threshold via cv2.adaptiveThreshold
      4. Inverted image
    """
    if not _HAS_ZXING_CPP:
        return None
    if region.ndim == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region
    if not gray.flags["C_CONTIGUOUS"]:
        gray = np.ascontiguousarray(gray)
    try:
        # Attempt 1: LocalAverage
        results = zxingcpp.read_barcodes(gray)
        for r in results:
            if r.text:
                return r.text

        # Attempt 2: GlobalHistogram
        try:
            results = zxingcpp.read_barcodes(
                gray, binarizer=zxingcpp.Binarizer.GlobalHistogram)
            for r in results:
                if r.text:
                    return r.text
        except (AttributeError, TypeError):
            pass

        # Attempt 3: adaptive threshold
        h, w = gray.shape[:2]
        if h >= 25 and w >= 25:
            bs = w // 10
            bs = bs + (bs % 2) - 1
            if bs >= 3:
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, bs, 10)
                results = zxingcpp.read_barcodes(
                    thresh, binarizer=zxingcpp.Binarizer.BoolCast)
                for r in results:
                    if r.text:
                        return r.text

        # Attempt 4: inverted
        inverted = cv2.bitwise_not(gray)
        results = zxingcpp.read_barcodes(inverted)
        for r in results:
            if r.text:
                return r.text
    except Exception:
        pass
    return None


# ── Main survey logic ─────────────────────────────────────────────

def survey_video(
    video_path: str,
    detector: _MNNDetector,
    sample_count: int,
) -> dict:
    """Run all candidate decoders on MNN-cropped regions from one video."""
    tag = Path(video_path).name
    frames = extract_frames(video_path, sample_count)
    if not frames:
        return {"video": tag, "error": "no frames extracted", "crops": 0}

    candidates = {
        "wechat_qr": decode_wechat_qr,
        "opencv_qr": decode_opencv_qr,
    }
    if _HAS_ZXING_CPP:
        candidates["zxing_cpp"] = decode_zxing_cpp
        candidates["zxing_cpp_multi"] = decode_zxing_cpp_multi

    # Collect crops via MNN detector
    crops = []  # list of (frame_idx, crop_bgr)
    for fi, frame in enumerate(frames):
        bboxes = detector.detect_bboxes(frame)
        img_h, img_w = frame.shape[:2]
        for bbox in bboxes:
            px0, py0, px1, py1 = _pad_bbox(*bbox, img_w, img_h)
            crop = frame[py0:py1, px0:px1]
            if crop.size > 0:
                crops.append((fi, crop))

    if not crops:
        return {
            "video": tag,
            "frames_sampled": len(frames),
            "crops": 0,
            "note": "MNN detected no QR in any sampled frame",
        }

    # Run each candidate on every crop
    results: dict[str, dict] = {}
    for cname, cfn in candidates.items():
        hits = 0
        times_ms = []
        decoded_set: set[int] = set()  # frame indices decoded successfully
        for fi, crop in crops:
            t0 = time.perf_counter()
            text = cfn(crop)
            elapsed = (time.perf_counter() - t0) * 1000
            times_ms.append(elapsed)
            if text:
                hits += 1
                decoded_set.add(fi)
        arr = np.array(times_ms) if times_ms else np.array([0.0])
        results[cname] = {
            "hits": hits,
            "total": len(crops),
            "hit_rate": round(hits / len(crops) * 100, 1) if crops else 0,
            "avg_ms": round(float(arr.mean()), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "decoded_frames": sorted(decoded_set),
        }

    # Compute "exclusive WeChatQR misses" — frames where WeChatQR
    # decodes but the candidate does not.
    wechat_set = set(results["wechat_qr"]["decoded_frames"])
    for cname in results:
        if cname == "wechat_qr":
            results[cname]["exclusive_miss_vs_wechat"] = []
            continue
        cset = set(results[cname]["decoded_frames"])
        exclusive = sorted(wechat_set - cset)
        results[cname]["exclusive_miss_vs_wechat"] = exclusive
        results[cname]["exclusive_miss_count"] = len(exclusive)

    # Also record inverse: frames where candidate decodes but
    # WeChatQR does not (bonus capability).
    for cname in results:
        if cname == "wechat_qr":
            results[cname]["bonus_vs_wechat"] = []
            continue
        cset = set(results[cname]["decoded_frames"])
        bonus = sorted(cset - wechat_set)
        results[cname]["bonus_vs_wechat"] = bonus
        results[cname]["bonus_count"] = len(bonus)

    # Strip decoded_frames from output (too verbose for summary)
    for cname in results:
        del results[cname]["decoded_frames"]

    return {
        "video": tag,
        "frames_sampled": len(frames),
        "crops": len(crops),
        "candidates": results,
    }


def format_markdown(all_results: list[dict], meta: dict) -> str:
    """Format survey results as a Markdown report."""
    lines = [
        "# M1.75 CPU Decoder Survey",
        "",
        f"**Platform**: {meta.get('platform', '?')}",
        f"**Python**: {meta.get('python', '?')}",
        f"**OpenCV**: {meta.get('opencv', '?')}",
        f"**MNN**: {meta.get('mnn', '?')}",
        f"**zxing-cpp**: {'available' if _HAS_ZXING_CPP else 'not installed'}",
        "",
    ]

    for vr in all_results:
        lines.append(f"## {vr['video']}")
        lines.append("")
        lines.append(f"- Frames sampled: {vr.get('frames_sampled', '?')}")
        lines.append(f"- Crops (MNN detections): {vr.get('crops', 0)}")
        if vr.get("error") or vr.get("note"):
            lines.append(f"- Note: {vr.get('error') or vr.get('note')}")
            lines.append("")
            continue

        cands = vr.get("candidates", {})
        if not cands:
            lines.append("")
            continue

        lines.append("")
        header = "| Decoder | Hits | Total | Hit Rate | Avg ms | P50 ms | P95 ms | Excl. Miss vs WeChatQR | Bonus vs WeChatQR |"
        sep = "|---------|------|-------|----------|--------|--------|--------|----------------------|-------------------|"
        lines.append(header)
        lines.append(sep)

        for cname, cs in cands.items():
            excl = cs.get("exclusive_miss_count", "—")
            bonus = cs.get("bonus_count", "—")
            lines.append(
                f"| {cname} | {cs['hits']} | {cs['total']} | "
                f"{cs['hit_rate']}% | {cs['avg_ms']} | "
                f"{cs['p50_ms']} | {cs['p95_ms']} | "
                f"{excl} | {bonus} |"
            )
        lines.append("")

    # Summary / recommendation
    lines.append("## Summary & Recommendation")
    lines.append("")
    lines.append("*(auto-generated — review before acting)*")
    lines.append("")

    # Aggregate across videos
    agg: dict[str, dict] = {}
    for vr in all_results:
        for cname, cs in vr.get("candidates", {}).items():
            if cname not in agg:
                agg[cname] = {"hits": 0, "total": 0, "times": [],
                              "excl_miss": 0, "bonus": 0}
            agg[cname]["hits"] += cs["hits"]
            agg[cname]["total"] += cs["total"]
            agg[cname]["times"].append(cs["avg_ms"])
            agg[cname]["excl_miss"] += cs.get("exclusive_miss_count", 0)
            agg[cname]["bonus"] += cs.get("bonus_count", 0)

    for cname, a in agg.items():
        rate = round(a["hits"] / a["total"] * 100, 1) if a["total"] else 0
        avg = round(sum(a["times"]) / len(a["times"]), 2) if a["times"] else 0
        lines.append(
            f"- **{cname}**: aggregate hit rate = {rate}% "
            f"({a['hits']}/{a['total']}), "
            f"avg decode = {avg} ms, "
            f"exclusive misses vs WeChatQR = {a['excl_miss']}, "
            f"bonus decodes = {a['bonus']}"
        )

    lines.append("")
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="M1.75 CPU decoder candidate survey")
    parser.add_argument(
        "--videos", "-v", nargs="*",
        help="Video files or directories to scan. "
             "Defaults to tests/fixtures/**/*.mp4",
    )
    parser.add_argument(
        "--sample-count", "-n", type=int, default=200,
        help="Max frames to sample per video (default: 200)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Write Markdown report to this path",
    )
    parser.add_argument(
        "--json", default=None,
        help="Write raw JSON results to this path",
    )
    parser.add_argument(
        "--mnn-backend", default="CPU",
        help="MNN backend for the detector (default: CPU)",
    )
    args = parser.parse_args()

    if not _HAS_MNN:
        print("ERROR: MNN not installed. Run inside the M1.75 container.", file=sys.stderr)
        sys.exit(1)

    detect_model = str(MNN_MODEL_DIR / "detect.mnn")
    if not Path(detect_model).exists():
        print(f"ERROR: detect model not found at {detect_model}", file=sys.stderr)
        sys.exit(1)

    # Discover videos
    video_paths: list[str] = []
    sources = args.videos or [str(PROJ_ROOT / "tests" / "fixtures")]
    for src in sources:
        p = Path(src)
        if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi"):
            video_paths.append(str(p))
        elif p.is_dir():
            video_paths.extend(
                str(f) for f in sorted(p.rglob("*.mp4"))
            )
    if not video_paths:
        print("ERROR: no video files found", file=sys.stderr)
        sys.exit(1)

    print("=" * 72)
    print("M1.75 CPU Decoder Survey")
    print("=" * 72)
    print(f"Platform:    {platform.platform()}")
    print(f"Python:      {sys.version.split()[0]}")
    print(f"OpenCV:      {cv2.__version__}")
    print(f"MNN:         {MNN.version()}")
    print(f"zxing-cpp:   {'yes' if _HAS_ZXING_CPP else 'no'}")
    print(f"Videos:      {len(video_paths)}")
    print(f"Samples/vid: {args.sample_count}")
    print()

    detector = _MNNDetector(detect_model, args.mnn_backend)
    all_results = []

    for vp in video_paths:
        print(f"  Surveying {Path(vp).name} ...", end="", flush=True)
        t0 = time.perf_counter()
        result = survey_video(vp, detector, args.sample_count)
        elapsed = time.perf_counter() - t0
        print(f" done ({elapsed:.1f}s, {result.get('crops', 0)} crops)")
        all_results.append(result)

    # Print summary table
    print()
    for vr in all_results:
        print(f"--- {vr['video']} ({vr.get('crops', 0)} crops) ---")
        for cname, cs in vr.get("candidates", {}).items():
            print(
                f"  {cname:20s}  "
                f"hit={cs['hit_rate']:5.1f}%  "
                f"avg={cs['avg_ms']:7.2f}ms  "
                f"p95={cs['p95_ms']:7.2f}ms  "
                f"excl_miss={cs.get('exclusive_miss_count', '—'):>3}  "
                f"bonus={cs.get('bonus_count', '—'):>3}"
            )
    print()

    meta = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "mnn": MNN.version(),
    }

    # Write outputs
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"meta": meta, "results": all_results}, f, indent=2)
        print(f"JSON saved: {out}")

    md = format_markdown(all_results, meta)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        print(f"Report saved: {out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
