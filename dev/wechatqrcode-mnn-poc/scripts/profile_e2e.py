"""End-to-end profile for M3 task 3.1.

Purpose
-------
M3 re-scoped from "batch acceleration" to "profile-driven end-to-end
optimisation" after the MNN batch probes (see §3.1a in README).  Before
touching any production code, we need a concrete breakdown of where
the ~67 ms/frame goes in the MNN path:

  frame I/O  →  downscale  →  preprocess  →  CNN detect  →
  bbox clamp/pad  →  crop  →  optional SR  →  zxing decode  →  LT

This script monkey-patches ``MNNQrDetector`` (non-invasively — only
affects the child process inside podman) to record per-stage timings
on every frame, then aggregates them into p50/p95/mean and a cumulative
breakdown.

Outputs
-------
JSON report at ``--output`` containing, for each sample:

  frames_scanned      : number of frames actually sent to detector
  frames_with_detect  : number of frames where CNN returned ≥ 1 bbox
  frames_with_decode  : number of frames where a QR string was decoded
  stages:
    cnn_detect_ms       : CNN inference only
    sr_ms               : SR inference (only on frames where SR was applied)
    sr_triggered_count  : how many frames triggered SR
    cpu_decode_ms       : zxing-cpp call time (including multi-binarization retries)
    total_detect_ms     : wall clock of MNNQrDetector.detect()
  stats               : DetectorRouter stats at end
  wall_clock_s        : total scan wall clock (frames/sec)

Running
-------
Designed to run inside podman.  See Containerfile.m3_profile.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path


def _install_profile_hooks(detector) -> dict:
    """Monkey-patch a fresh ``MNNQrDetector`` to record per-stage timings.

    Returns a dict that the patches will fill with timing lists.
    We patch *bound methods* on the instance so no class state leaks,
    and we chain to the original methods so behaviour is preserved.

    Stages recorded:
      - cnn_detect_ms: time inside ``_run_detector``
      - sr_ms: time inside ``_run_sr`` (only counted when invoked)
      - cpu_decode_ms: time inside ``_cpu_decode``
      - total_detect_ms: time inside ``detect`` (top-level)
    """
    buckets = {
        "cnn_detect_ms": [],
        "sr_ms": [],
        "cpu_decode_ms": [],
        "total_detect_ms": [],
        "sr_triggered_count": 0,
        "frames_with_detect": 0,
        "frames_with_decode": 0,
        "frames_scanned": 0,
    }

    orig_run_detector = detector._run_detector
    orig_run_sr = detector._run_sr
    orig_cpu_decode = detector._cpu_decode
    orig_detect = detector.detect

    def patched_run_detector(frame, detect_sess):
        t0 = time.perf_counter()
        try:
            result = orig_run_detector(frame, detect_sess)
            return result
        finally:
            buckets["cnn_detect_ms"].append(
                (time.perf_counter() - t0) * 1000.0,
            )

    def patched_run_sr(crop, sr_sess):
        t0 = time.perf_counter()
        try:
            return orig_run_sr(crop, sr_sess)
        finally:
            buckets["sr_ms"].append((time.perf_counter() - t0) * 1000.0)
            buckets["sr_triggered_count"] += 1

    def patched_cpu_decode(region):
        t0 = time.perf_counter()
        try:
            return orig_cpu_decode(region)
        finally:
            buckets["cpu_decode_ms"].append(
                (time.perf_counter() - t0) * 1000.0,
            )

    def patched_detect(frame):
        buckets["frames_scanned"] += 1
        t0 = time.perf_counter()
        try:
            result = orig_detect(frame)
            # Classify: did CNN return anything vs did decode succeed?
            # We can't tell "detect but no decode" from ``result`` alone,
            # so approximate via cnn_detect_ms growth vs previous call.
            if result.text is not None:
                buckets["frames_with_decode"] += 1
            return result
        finally:
            buckets["total_detect_ms"].append(
                (time.perf_counter() - t0) * 1000.0,
            )

    detector._run_detector = patched_run_detector
    detector._run_sr = patched_run_sr
    detector._cpu_decode = patched_cpu_decode
    detector.detect = patched_detect
    return buckets


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    a = sorted(values)
    return {
        "n": len(a),
        "p50": round(statistics.median(a), 3),
        "p95": round(a[int(len(a) * 0.95)] if len(a) > 1 else a[0], 3),
        "p99": round(a[int(len(a) * 0.99)] if len(a) > 1 else a[0], 3),
        "mean": round(statistics.mean(a), 3),
        "sum_ms": round(sum(a), 1),
        "min": round(min(a), 3),
        "max": round(max(a), 3),
    }


def _profile_one_sample(
    video_path: Path,
    expected_sha256: str | None,
    workers: int,
    use_mnn: bool,
    detect_isolation: str = "off",
) -> dict:
    """Run one end-to-end decode, with profile hooks if MNN is on."""
    # Import inside the function so the patches don't affect other samples.
    import hashlib

    from qrstream.decoder import extract_qr_from_video, decode_blocks
    from qrstream.detector.router import DetectorRouter

    # Build a router up-front so we can reach the underlying MNN instance
    # AFTER it's been lazily constructed.  We do this by patching
    # DetectorRouter._get_mnn_detector to install hooks the first time
    # it returns a fresh MNNQrDetector.
    hooks_buckets_ref: list[dict] = []

    orig_get = DetectorRouter._get_mnn_detector

    def patched_get(self):
        inst = orig_get(self)
        if inst is not None and not getattr(inst, "_profile_hooked", False):
            buckets = _install_profile_hooks(inst)
            inst._profile_hooked = True
            hooks_buckets_ref.append(buckets)
        return inst

    DetectorRouter._get_mnn_detector = patched_get
    try:
        t0 = time.perf_counter()
        blocks = extract_qr_from_video(
            str(video_path),
            sample_rate=0,
            verbose=False,
            workers=workers,
            use_mnn=use_mnn,
            detect_isolation=detect_isolation,
        )
        extract_wall = time.perf_counter() - t0

        t1 = time.perf_counter()
        payload = decode_blocks(blocks)
        decode_wall = time.perf_counter() - t1
    finally:
        DetectorRouter._get_mnn_detector = orig_get

    sha_ok = None
    payload_sha256 = None
    if payload is not None:
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256:
            sha_ok = (payload_sha256 == expected_sha256)

    report = {
        "video_path": str(video_path),
        "use_mnn": use_mnn,
        "workers": workers,
        "extract_wall_s": round(extract_wall, 3),
        "decode_wall_s": round(decode_wall, 3),
        "total_wall_s": round(extract_wall + decode_wall, 3),
        "payload_bytes": len(payload) if payload else 0,
        "payload_sha256": payload_sha256,
        "sha256_match": sha_ok,
        "num_unique_blocks_from_extract": len(blocks),
    }

    if hooks_buckets_ref:
        buckets = hooks_buckets_ref[0]
        report["stages"] = {
            "cnn_detect_ms": _stats(buckets["cnn_detect_ms"]),
            "sr_ms": _stats(buckets["sr_ms"]),
            "cpu_decode_ms": _stats(buckets["cpu_decode_ms"]),
            "total_detect_ms": _stats(buckets["total_detect_ms"]),
        }
        report["counters"] = {
            "frames_scanned": buckets["frames_scanned"],
            "frames_with_decode": buckets["frames_with_decode"],
            "sr_triggered_count": buckets["sr_triggered_count"],
        }
        # Cumulative breakdown: what fraction of wall clock is each stage?
        total_detect_sum = sum(buckets["total_detect_ms"])
        report["cumulative_ms"] = {
            "cnn_detect": round(sum(buckets["cnn_detect_ms"]), 1),
            "sr": round(sum(buckets["sr_ms"]), 1),
            "cpu_decode": round(sum(buckets["cpu_decode_ms"]), 1),
            "total_detect": round(total_detect_sum, 1),
        }

    return report


def _compute_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--samples-root", type=Path, required=True,
        help="Path to tests/fixtures or equivalent containing real-phone-v3 / v4",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--use-mnn", action="store_true",
        help="Profile MNN path (default: OpenCV baseline)",
    )
    ap.add_argument(
        "--samples", default="v061,v070,v073-10kB,v073-100kB,v073-300kB",
        help="Comma-separated sample names",
    )
    ap.add_argument(
        "--extra-sample", action="append", default=[],
        help="Ad-hoc sample, format NAME=PATH_TO_MP4 (repeatable). "
             "No expected input.bin; records payload_sha256 instead.",
    )
    ap.add_argument(
        "--detect-isolation", choices=["on", "off"], default="on",
        help="Pass to extract_qr_from_video.  Default 'on' for safety "
             "on host (OpenCV WeChatQRCode is known to SIGSEGV on some "
             "frames); use 'off' inside podman where isolation adds "
             "fork overhead and the sample set is already vetted.",
    )
    args = ap.parse_args()

    # Build sample → (mp4, input.bin) map
    sample_map = {
        "v061": (
            args.samples_root / "real-phone-v3" / "v061.mp4",
            args.samples_root / "real-phone-v3" / "v061.input.bin",
        ),
        "v070": (
            args.samples_root / "real-phone-v3" / "v070.mp4",
            args.samples_root / "real-phone-v3" / "v070.input.bin",
        ),
        "v073-10kB": (
            args.samples_root / "real-phone-v4" / "v073-10kB.mp4",
            args.samples_root / "real-phone-v4" / "v073-10kB.input.bin",
        ),
        "v073-100kB": (
            args.samples_root / "real-phone-v4" / "v073-100kB.mp4",
            args.samples_root / "real-phone-v4" / "v073-100kB.input.bin",
        ),
        "v073-300kB": (
            args.samples_root / "real-phone-v4" / "v073-300kB.mp4",
            args.samples_root / "real-phone-v4" / "v073-300kB.input.bin",
        ),
        # Auxiliary sample: IMG_9425.MOV lives outside tests/fixtures
        # (it's the benchmark reference from the M1.5 report).  The
        # probe will run it when ``--samples`` includes ``IMG_9425``
        # and ``--extra-sample IMG_9425=/abs/path.MOV`` is provided.
    }

    # Allow extra ad-hoc samples passed as --extra-sample NAME=PATH
    for extra in getattr(args, "extra_sample", []) or []:
        if "=" not in extra:
            continue
        name, path = extra.split("=", 1)
        p = Path(path)
        # No expected input.bin for extra samples — they use
        # self-consistency (sha256_match reported as None).
        sample_map[name] = (p, Path("/nonexistent"))

    requested = [s.strip() for s in args.samples.split(",") if s.strip()]

    report = {
        "config": {
            "workers": args.workers,
            "use_mnn": args.use_mnn,
            "samples_requested": requested,
        },
        "samples": {},
    }

    for name in requested:
        if name not in sample_map:
            report["samples"][name] = {"error": f"unknown sample {name}"}
            continue
        mp4, input_bin = sample_map[name]
        if not mp4.exists():
            report["samples"][name] = {"error": f"missing {mp4}"}
            continue
        expected_sha = (
            _compute_sha256(input_bin) if input_bin.exists() else None
        )
        try:
            entry = _profile_one_sample(
                mp4, expected_sha, args.workers, args.use_mnn,
                detect_isolation=args.detect_isolation,
            )
        except Exception as e:
            entry = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        report["samples"][name] = entry
        print(
            f"[profile] {name}: "
            f"wall={entry.get('total_wall_s', '?')}s  "
            f"sha_ok={entry.get('sha256_match')}  "
            f"frames={entry.get('counters', {}).get('frames_scanned', '?')}",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
