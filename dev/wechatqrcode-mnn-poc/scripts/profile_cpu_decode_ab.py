"""Profile the **production** _cpu_decode code path after M3 §3.2-AB.

Unlike ``profile_cpu_decode_attempts.py`` (which inlines a copy of the
M1.75 4-attempt chain for per-attempt timing), this script keeps the
production ``_decode_zxing_cpp`` untouched and only records:

  - Whether each ``_cpu_decode`` call succeeded
  - Time spent per call
  - Whether attempt 3's bbox gate was opened (inspect crop dims)

Purpose: verify that the M3 §3.2-AB changes (remove attempt 4, gate
attempt 3 on ``min(h,w) >= 80``) actually show up in the production
call path on the same fixtures.
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


def _install_call_hooks(detector):
    """Wrap the detector's ``_cpu_decode`` to record call stats only.

    The underlying ``_decode_zxing_cpp`` is **not** patched — we
    observe whatever the production code does.
    """
    orig = detector._cpu_decode
    buckets = {
        "call_ms": [],
        "call_success_ms": [],
        "call_fail_ms": [],
        "small_crop_count": 0,   # min(h,w) < 80: attempt 3 skipped
        "large_crop_count": 0,   # min(h,w) >= 80: attempt 3 eligible
        "total_crops": 0,
        "successes": 0,
    }

    def patched(region):
        buckets["total_crops"] += 1
        if region is not None and hasattr(region, "shape"):
            if region.ndim >= 2:
                h, w = region.shape[:2]
                if min(h, w) >= 80:
                    buckets["large_crop_count"] += 1
                else:
                    buckets["small_crop_count"] += 1
        t0 = time.perf_counter()
        try:
            result = orig(region)
            return result
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0
            buckets["call_ms"].append(elapsed)
            if result:  # noqa: F821 — captured by closure
                buckets["call_success_ms"].append(elapsed)
                buckets["successes"] += 1
            else:
                buckets["call_fail_ms"].append(elapsed)

    detector._cpu_decode = patched
    return buckets


def _profile(video, expected_sha, workers):
    from qrstream.decoder import extract_qr_from_video, decode_blocks
    from qrstream.detector.router import DetectorRouter

    hooks_ref = []
    orig_get = DetectorRouter._get_mnn_detector

    def patched_get(self):
        inst = orig_get(self)
        if inst is not None and not getattr(inst, "_ab_hooked", False):
            b = _install_call_hooks(inst)
            inst._ab_hooked = True
            hooks_ref.append(b)
        return inst

    DetectorRouter._get_mnn_detector = patched_get
    try:
        t0 = time.perf_counter()
        blocks = extract_qr_from_video(
            str(video),
            sample_rate=0,
            verbose=False,
            workers=workers,
            use_mnn=True,
            detect_isolation="off",
        )
        wall = time.perf_counter() - t0
        payload = decode_blocks(blocks)
    finally:
        DetectorRouter._get_mnn_detector = orig_get

    sha = (hashlib.sha256(payload).hexdigest() == expected_sha
           if (payload and expected_sha) else None)

    out = {
        "video": str(video),
        "wall_s": round(wall, 3),
        "sha_ok": sha,
    }
    if hooks_ref:
        b = hooks_ref[0]
        out["total_crops"] = b["total_crops"]
        out["success_count"] = b["successes"]
        out["success_rate"] = round(
            b["successes"] / max(1, b["total_crops"]), 3,
        )
        out["small_crop_count"] = b["small_crop_count"]   # attempt3 skipped
        out["large_crop_count"] = b["large_crop_count"]   # attempt3 eligible
        out["call_ms"] = _stats(b["call_ms"])
        out["call_success_ms"] = _stats(b["call_success_ms"])
        out["call_fail_ms"] = _stats(b["call_fail_ms"])
    return out


def _sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--samples", default="v061,v070,v073-10kB,v073-100kB,v073-300kB",
    )
    args = ap.parse_args()

    sample_map = {
        "v061": ("real-phone-v3/v061",),
        "v070": ("real-phone-v3/v070",),
        "v073-10kB": ("real-phone-v4/v073-10kB",),
        "v073-100kB": ("real-phone-v4/v073-100kB",),
        "v073-300kB": ("real-phone-v4/v073-300kB",),
    }

    report = {"samples": {}}
    for name in [s.strip() for s in args.samples.split(",") if s.strip()]:
        if name not in sample_map:
            continue
        base = args.samples_root / sample_map[name][0]
        mp4 = base.with_suffix(".mp4")
        bin_ = base.parent / (base.name + ".input.bin")
        if not mp4.exists():
            report["samples"][name] = {"error": f"missing {mp4}"}
            continue
        expected = _sha256(bin_) if bin_.exists() else None
        try:
            entry = _profile(mp4, expected, args.workers)
        except Exception as e:
            entry = {"error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()}
        report["samples"][name] = entry
        print(
            f"[ab] {name}: crops={entry.get('total_crops','?')} "
            f"hit={entry.get('success_rate','?')} "
            f"mean={entry.get('call_ms',{}).get('mean','?')}ms "
            f"small={entry.get('small_crop_count','?')}/"
            f"{entry.get('total_crops','?')}",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
