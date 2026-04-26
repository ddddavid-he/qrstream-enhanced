"""Probe MNN detector confidence vs decode-success correlation.

Purpose (M3 §3.2-C)
-------------------
The current ``MNNQrDetector._run_detector`` accepts every bbox with
``confidence > 1e-5``.  MNN's Caffe DetectionOutput layer already
applies the prototxt's ``confidence_threshold: 0.2`` internally, so
all bboxes fed to ``_cpu_decode`` have ``conf >= 0.2``.  But the
profile in §3.1 showed many crops walk the full attempt chain only
to fail — these are likely false-positive bboxes from the SSD whose
confidence sits in the [0.2, 0.5] band.

This probe measures, for every bbox MNN emits during a real-video
scan, whether the subsequent ``_cpu_decode`` succeeds, then bins by
confidence.  The goal is to identify a threshold ``T`` such that
crops with ``conf < T`` overwhelmingly fail, so we can short-circuit
them and save the ~15 ms / failed-crop cost.

Output
------
Per-sample histogram of (confidence_bin, attempts, successes) so we
can pick T from data, not assumption.  For each sample also reports
the "would-be saved cost" for several candidate thresholds:
0.30 / 0.40 / 0.50 / 0.60 / 0.70.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np


def _install_conf_hooks(detector):
    """Patch detector to record (confidence, decode_success?, decode_ms)
    for every bbox MNN emits during the run.

    We can't peek at conf inside ``_run_detector`` without changing
    its return contract, so we wrap ``_run_detector`` to attach the
    raw confidence to each bbox via a sidecar list, and then wrap
    ``_cpu_decode`` to consume from the same sidecar.  This keeps
    the production path observable without changing its semantics.

    The chosen approach is non-invasive: we run the original
    ``_run_detector`` to get bboxes, then *re-parse* the same MNN
    output tensor to recover confidences.  This costs one extra
    output-tensor parse per call (negligible vs total decode cost).
    """
    orig_run_detector = detector._run_detector
    orig_cpu_decode = detector._cpu_decode

    # Per-call sidecar: list of confidences in the order ``detect``
    # consumes bboxes.  Filled by the patched _run_detector, drained
    # by _cpu_decode.
    sidecar = {"confs": [], "idx": 0}

    buckets = {
        # (conf_bin, success_count, attempt_count, sum_ms)
        "by_bin": defaultdict(lambda: {"attempts": 0, "successes": 0,
                                       "sum_ms": 0.0}),
        "all_records": [],   # (conf, success, ms) tuples
        "no_conf_calls": 0,  # decode called but sidecar empty (shouldn't happen)
    }

    def patched_run_detector(frame, detect_sess):
        # Re-parse the MNN output tensor to recover confidences in
        # the same order as the bboxes _run_detector returns.
        import MNN
        import cv2 as _cv2
        bboxes = orig_run_detector(frame, detect_sess)

        # Recover confs: re-walk the same logic _run_detector did.
        # The cheap-and-dirty way: since _run_detector iterates
        # output_data[row], and accepts only class_id==1 & conf>1e-5,
        # we replicate just the conf collection by peeking at the
        # session's output tensor right after run.
        # (We rely on the fact that _run_detector finished a runSession
        # before returning, so the output is still the latest.)
        try:
            interpreter, session = detect_sess
            output = interpreter.getSessionOutput(session, "detection_output")
            shape = output.getShape()
            if len(shape) >= 4:
                tmp = MNN.Tensor(
                    shape, MNN.Halide_Type_Float,
                    np.zeros(shape, dtype=np.float32),
                    MNN.Tensor_DimensionType_Caffe,
                )
                output.copyToHostTensor(tmp)
                arr = np.array(tmp.getData(), dtype=np.float32)
                num = shape[2]
                dim = shape[3]
                arr = arr[:num * dim].reshape(num, dim)
                confs = []
                for r in arr:
                    if dim == 7:
                        cls, c = r[1], r[2]
                    else:
                        cls, c = r[0], r[1]
                    if cls == 1 and c > 1e-5:
                        confs.append(float(c))
                # Match length to bboxes (defensive — should be equal)
                sidecar["confs"] = confs[:len(bboxes)]
                sidecar["idx"] = 0
            else:
                sidecar["confs"] = []
                sidecar["idx"] = 0
        except Exception:
            sidecar["confs"] = []
            sidecar["idx"] = 0
        return bboxes

    def patched_cpu_decode(region):
        if sidecar["idx"] < len(sidecar["confs"]):
            conf = sidecar["confs"][sidecar["idx"]]
            sidecar["idx"] += 1
        else:
            buckets["no_conf_calls"] += 1
            conf = -1.0  # unknown

        t0 = time.perf_counter()
        try:
            result = orig_cpu_decode(region)
            return result
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0
            success = bool(result)  # noqa: F821 — captured by closure
            buckets["all_records"].append((conf, success, elapsed))
            if conf >= 0:
                # Bin by 0.05 increments
                bin_lo = int(conf * 20) / 20  # floor to nearest 0.05
                key = round(bin_lo, 2)
                bd = buckets["by_bin"][key]
                bd["attempts"] += 1
                if success:
                    bd["successes"] += 1
                bd["sum_ms"] += elapsed

    detector._run_detector = patched_run_detector
    detector._cpu_decode = patched_cpu_decode
    return buckets


def _profile(video, expected_sha, workers, isolation):
    from qrstream.decoder import extract_qr_from_video, decode_blocks
    from qrstream.detector.router import DetectorRouter

    hooks_ref = []
    orig_get = DetectorRouter._get_mnn_detector

    def patched_get(self):
        inst = orig_get(self)
        if inst is not None and not getattr(inst, "_conf_hooked", False):
            b = _install_conf_hooks(inst)
            inst._conf_hooked = True
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
            detect_isolation=isolation,
        )
        wall = time.perf_counter() - t0
        payload = decode_blocks(blocks)
    finally:
        DetectorRouter._get_mnn_detector = orig_get

    sha_ok = (hashlib.sha256(payload).hexdigest() == expected_sha
              if (payload and expected_sha) else None)

    out = {
        "video": str(video),
        "wall_s": round(wall, 3),
        "sha_ok": sha_ok,
    }

    if hooks_ref:
        b = hooks_ref[0]
        records = b["all_records"]
        out["total_decode_calls"] = len(records)
        out["no_conf_calls"] = b["no_conf_calls"]

        # Histogram by 0.05-confidence bin
        bins = sorted(b["by_bin"].keys())
        hist = []
        for k in bins:
            v = b["by_bin"][k]
            hist.append({
                "bin_lo": k,
                "attempts": v["attempts"],
                "successes": v["successes"],
                "hit_rate": round(v["successes"] / max(1, v["attempts"]), 3),
                "mean_ms": round(v["sum_ms"] / max(1, v["attempts"]), 2),
                "sum_ms": round(v["sum_ms"], 1),
            })
        out["histogram"] = hist

        # "What if we threshold at T?" simulator
        scenarios = []
        for T in [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
            kept = [r for r in records if r[0] >= T]
            dropped = [r for r in records if 0 <= r[0] < T]
            kept_hits = sum(1 for r in kept if r[1])
            dropped_hits = sum(1 for r in dropped if r[1])
            saved_ms = sum(r[2] for r in dropped)
            kept_ms = sum(r[2] for r in kept)
            scenarios.append({
                "threshold": T,
                "kept_calls": len(kept),
                "dropped_calls": len(dropped),
                "kept_successes": kept_hits,
                "dropped_successes": dropped_hits,  # missed hits we'd lose
                "saved_decode_ms": round(saved_ms, 1),
                "kept_decode_ms": round(kept_ms, 1),
                "lost_hit_rate": round(
                    dropped_hits / max(1, kept_hits + dropped_hits), 4,
                ),
            })
        out["threshold_scenarios"] = scenarios

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
    ap.add_argument("--detect-isolation", default="on")
    ap.add_argument(
        "--samples", default="v061,v070,v073-10kB,v073-100kB,v073-300kB",
    )
    ap.add_argument("--extra-sample", action="append", default=[])
    args = ap.parse_args()

    sample_map = {
        "v061": (args.samples_root / "real-phone-v3" / "v061.mp4",
                 args.samples_root / "real-phone-v3" / "v061.input.bin"),
        "v070": (args.samples_root / "real-phone-v3" / "v070.mp4",
                 args.samples_root / "real-phone-v3" / "v070.input.bin"),
        "v073-10kB": (args.samples_root / "real-phone-v4" / "v073-10kB.mp4",
                      args.samples_root / "real-phone-v4" / "v073-10kB.input.bin"),
        "v073-100kB": (args.samples_root / "real-phone-v4" / "v073-100kB.mp4",
                       args.samples_root / "real-phone-v4" / "v073-100kB.input.bin"),
        "v073-300kB": (args.samples_root / "real-phone-v4" / "v073-300kB.mp4",
                       args.samples_root / "real-phone-v4" / "v073-300kB.input.bin"),
    }
    for extra in args.extra_sample:
        if "=" not in extra:
            continue
        n, p = extra.split("=", 1)
        sample_map[n] = (Path(p), Path("/nonexistent"))

    report = {"samples": {}}
    for name in [s.strip() for s in args.samples.split(",") if s.strip()]:
        if name not in sample_map:
            continue
        mp4, bin_ = sample_map[name]
        if not mp4.exists():
            report["samples"][name] = {"error": f"missing {mp4}"}
            continue
        expected = _sha256(bin_) if bin_.exists() else None
        try:
            entry = _profile(mp4, expected, args.workers,
                             args.detect_isolation)
        except Exception as e:
            entry = {"error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()}
        report["samples"][name] = entry
        # Quick stderr summary
        if "histogram" in entry:
            total = entry["total_decode_calls"]
            hits = sum(b["successes"] for b in entry["histogram"])
            print(f"[conf] {name}: calls={total} hits={hits} "
                  f"hit_rate={hits/max(1,total):.3f}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
