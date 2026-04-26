"""Drill-down profile: which zxing-cpp attempt succeeds / fails for each crop?

The baseline profile showed ``cpu_decode_ms`` mean=42 ms (vs M1.75's
claimed avg=17 ms).  The 4-attempt binarizer chain could explain this
if a large fraction of crops walk all 4 attempts before giving up.
This script monkey-patches ``MNNQrDetector._decode_zxing_cpp`` to
count attempts and record timings per-attempt, so we know exactly
where the time goes.

Output
------
For each sample, per-attempt stats:
  attempt_1_LocalAverage : {n, ms_stats}
  attempt_2_GlobalHistogram : {...}
  attempt_3_AdaptiveThresh : {...}
  attempt_4_Inverted : {...}
  decode_success_at_attempt : {1: count, 2: count, 3: count, 4: count}
  decode_miss_all_4 : count

Plus global:
  total_crops
  successful_crops
  all_4_fail_crops  (== total_crops - successful_crops)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import traceback
from pathlib import Path


def _stats(values):
    if not values:
        return {"n": 0}
    a = sorted(values)
    return {
        "n": len(a),
        "p50": round(statistics.median(a), 3),
        "p95": round(a[int(len(a) * 0.95)] if len(a) > 1 else a[0], 3),
        "mean": round(statistics.mean(a), 3),
        "sum_ms": round(sum(a), 1),
        "max": round(max(a), 3),
    }


def _install_decode_hooks(detector) -> dict:
    """Monkey-patch ``_decode_zxing_cpp`` with per-attempt accounting.

    Re-implements the 4-step chain inline so we can time each step
    separately.  This must match the production logic byte-for-byte
    (apart from the timing hooks).
    """
    import cv2
    import numpy as np
    try:
        import zxingcpp as _zx  # type: ignore
    except ImportError:
        _zx = None

    buckets = {
        "attempt_1_LocalAverage": [],
        "attempt_2_GlobalHistogram": [],
        "attempt_3_AdaptiveThresh": [],
        "attempt_4_Inverted": [],
        "success_at_attempt": {1: 0, 2: 0, 3: 0, 4: 0},
        "miss_all_4": 0,
        "total_crops": 0,
        "prep_ms": [],  # grayscale + contiguous
    }

    def patched_decode(region):
        buckets["total_crops"] += 1
        if _zx is None:
            return None
        try:
            # Prep
            t_prep = time.perf_counter()
            if region.ndim == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            elif region.ndim == 2:
                gray = region
            else:
                return None
            if not gray.flags["C_CONTIGUOUS"]:
                gray = np.ascontiguousarray(gray)
            buckets["prep_ms"].append(
                (time.perf_counter() - t_prep) * 1000.0,
            )

            # Attempt 1
            t0 = time.perf_counter()
            results = _zx.read_barcodes(gray)
            buckets["attempt_1_LocalAverage"].append(
                (time.perf_counter() - t0) * 1000.0,
            )
            for r in results:
                if r.text:
                    buckets["success_at_attempt"][1] += 1
                    return r.text

            # Attempt 2
            try:
                t0 = time.perf_counter()
                results = _zx.read_barcodes(
                    gray, binarizer=_zx.Binarizer.GlobalHistogram,
                )
                buckets["attempt_2_GlobalHistogram"].append(
                    (time.perf_counter() - t0) * 1000.0,
                )
                for r in results:
                    if r.text:
                        buckets["success_at_attempt"][2] += 1
                        return r.text
            except (AttributeError, TypeError):
                pass

            # Attempt 3
            h, w = gray.shape[:2]
            if h >= 25 and w >= 25:
                bs = w // 10
                bs = bs + (bs % 2) - 1
                if bs >= 3:
                    t0 = time.perf_counter()
                    thresh = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, bs, 10,
                    )
                    results = _zx.read_barcodes(
                        thresh, binarizer=_zx.Binarizer.BoolCast,
                    )
                    buckets["attempt_3_AdaptiveThresh"].append(
                        (time.perf_counter() - t0) * 1000.0,
                    )
                    for r in results:
                        if r.text:
                            buckets["success_at_attempt"][3] += 1
                            return r.text

            # Attempt 4
            t0 = time.perf_counter()
            inverted = cv2.bitwise_not(gray)
            results = _zx.read_barcodes(inverted)
            buckets["attempt_4_Inverted"].append(
                (time.perf_counter() - t0) * 1000.0,
            )
            for r in results:
                if r.text:
                    buckets["success_at_attempt"][4] += 1
                    return r.text

            buckets["miss_all_4"] += 1
            return None
        except Exception:
            return None

    detector._cpu_decode = patched_decode
    return buckets


def _profile_sample(video_path, expected_sha, workers, isolation="off") -> dict:
    from qrstream.decoder import extract_qr_from_video, decode_blocks
    from qrstream.detector.router import DetectorRouter

    hooks_ref = []
    orig_get = DetectorRouter._get_mnn_detector

    def patched_get(self):
        inst = orig_get(self)
        if inst is not None and not getattr(inst, "_decode_hooked", False):
            b = _install_decode_hooks(inst)
            inst._decode_hooked = True
            hooks_ref.append(b)
        return inst

    DetectorRouter._get_mnn_detector = patched_get
    try:
        t0 = time.perf_counter()
        blocks = extract_qr_from_video(
            str(video_path),
            sample_rate=0,
            verbose=False,
            workers=workers,
            use_mnn=True,
            detect_isolation=isolation,
        )
        extract_wall = time.perf_counter() - t0
        payload = decode_blocks(blocks)
    finally:
        DetectorRouter._get_mnn_detector = orig_get

    sha_ok = None
    if expected_sha and payload is not None:
        sha_ok = hashlib.sha256(payload).hexdigest() == expected_sha

    report = {
        "video": str(video_path),
        "extract_wall_s": round(extract_wall, 3),
        "sha256_match": sha_ok,
    }

    if hooks_ref:
        b = hooks_ref[0]
        report["total_crops"] = b["total_crops"]
        report["success_at_attempt"] = b["success_at_attempt"]
        report["miss_all_4"] = b["miss_all_4"]
        report["success_rate"] = round(
            sum(b["success_at_attempt"].values()) / max(1, b["total_crops"]), 3,
        )
        report["prep_ms"] = _stats(b["prep_ms"])
        report["attempts"] = {
            "1_LocalAverage": _stats(b["attempt_1_LocalAverage"]),
            "2_GlobalHistogram": _stats(b["attempt_2_GlobalHistogram"]),
            "3_AdaptiveThresh": _stats(b["attempt_3_AdaptiveThresh"]),
            "4_Inverted": _stats(b["attempt_4_Inverted"]),
        }
        # "avg ms per crop" matching M1.75 survey's definition
        total_attempt_ms = (
            sum(b["attempt_1_LocalAverage"]) +
            sum(b["attempt_2_GlobalHistogram"]) +
            sum(b["attempt_3_AdaptiveThresh"]) +
            sum(b["attempt_4_Inverted"])
        )
        report["total_attempt_ms_sum"] = round(total_attempt_ms, 1)
        report["avg_ms_per_crop"] = (
            round(total_attempt_ms / b["total_crops"], 3)
            if b["total_crops"] else 0
        )
    return report


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--detect-isolation", default="off")
    ap.add_argument(
        "--samples", default="v061,v070,v073-10kB,v073-100kB,v073-300kB",
    )
    ap.add_argument("--extra-sample", action="append", default=[])
    args = ap.parse_args()

    sample_map = {
        "v061": ("real-phone-v3/v061", ),
        "v070": ("real-phone-v3/v070", ),
        "v073-10kB": ("real-phone-v4/v073-10kB", ),
        "v073-100kB": ("real-phone-v4/v073-100kB", ),
        "v073-300kB": ("real-phone-v4/v073-300kB", ),
    }
    # Extra samples live outside the fixtures tree.  Format: NAME=PATH
    extra_direct = {}
    for extra in args.extra_sample:
        if "=" not in extra:
            continue
        n, p = extra.split("=", 1)
        extra_direct[n] = Path(p)

    requested = [s.strip() for s in args.samples.split(",") if s.strip()]
    report = {"samples": {}}

    for name in requested:
        if name in extra_direct:
            mp4 = extra_direct[name]
            bin_ = Path("/nonexistent")  # no expected SHA for extras
        elif name in sample_map:
            base = args.samples_root / sample_map[name][0]
            mp4 = base.with_suffix(".mp4")
            bin_ = base.with_suffix("").parent / (base.name + ".input.bin")
        else:
            continue
        if not mp4.exists():
            report["samples"][name] = {"error": f"missing {mp4}"}
            continue
        expected = _sha256(bin_) if bin_.exists() else None
        try:
            entry = _profile_sample(mp4, expected, args.workers,
                                    isolation=args.detect_isolation)
        except Exception as e:
            entry = {"error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()}
        report["samples"][name] = entry
        print(
            f"[decode-profile] {name}: "
            f"crops={entry.get('total_crops', '?')}  "
            f"hit={entry.get('success_rate', '?')}  "
            f"avg={entry.get('avg_ms_per_crop', '?')}ms  "
            f"miss={entry.get('miss_all_4', '?')}",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
