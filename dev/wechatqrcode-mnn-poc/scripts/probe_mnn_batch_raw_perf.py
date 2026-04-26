"""Path-3 feasibility probe: measure batched *raw tensor* throughput.

Background
----------
``probe_mnn_batch_output_v2.py`` proved that MNN 3.5's ``DetectionOutput``
operator silently drops all detections when batch_size > 1.  The
fallback plan (README §关键风险 1, "网络只输出 raw tensor, 在宿主代码中自己实现
decode / NMS / bbox 还原") requires two independent pieces of evidence
before we can commit to implementing it:

  (E1) The *raw* intermediate tensors that feed DetectionOutput —
       ``mbox_loc``, ``mbox_conf_flatten``, ``mbox_priorbox`` —
       must themselves be obtainable under batch_size > 1.
  (E2) Running the model up to those tensors (skipping DetectionOutput)
       at batch_size N must be measurably faster per-frame than
       running DetectionOutput-free batch=1 N times sequentially.

This probe answers both.  It does NOT implement NMS/decode — we only
want the speed signal.  If E2 shows <1.5× per-frame speedup at batch=4,
path-3 is dead in the water and we fall back to path-2 (new model).

Output
------
JSON report written to ``--output``.  Key fields per batch_size:

  raw_tensor_shapes         : shapes of mbox_loc/conf/priorbox
  detection_output_ok       : whether DetectionOutput still returns
                              something (should be False for batch>1)
  per_frame_ms_p50          : median per-frame inference latency
  per_frame_ms_mean         : mean per-frame inference latency
  speedup_vs_batch1         : per-frame speedup relative to batch=1

``--backend`` lets us probe CPU and (on Apple) Metal.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

try:
    import MNN  # type: ignore
except ImportError:
    print("ERROR: MNN not installed.", file=sys.stderr)
    sys.exit(2)


# Fixed spatial size keeps batch as the only independent variable.
# Using 384×384 (upstream SSD default) keeps this measurement
# comparable to the M0 bench.
_H, _W = 384, 384

# Warmup + timed iterations per batch size.  Small numbers deliberately:
# we want a quick perf signal, not a benchmark paper.
_WARMUP = 3
_ITERS = 20


def _make_batch(batch_size: int) -> np.ndarray:
    """Build a (B, 1, H, W) float32 input with mild random noise.

    Content doesn't matter for perf — SSD kernels don't shortcut on
    zero input because of BatchNorm scale/shift, but we add noise to
    be safe against any potential zero-skip optimisations in MNN.
    """
    rng = np.random.default_rng(42)
    frames = rng.random(
        size=(batch_size, 1, _H, _W), dtype=np.float32,
    )
    return np.ascontiguousarray(frames)


def _probe_batch(
    model_path: Path,
    backend: str,
    batch_size: int,
) -> dict:
    """Build a fresh session; fetch raw + DetectionOutput tensors;
    time the inference.
    """
    interp = MNN.Interpreter(str(model_path))
    session = interp.createSession({"backend": backend})

    inp = interp.getSessionInput(session)
    interp.resizeTensor(inp, (batch_size, 1, _H, _W))
    interp.resizeSession(session)

    # Fetch all four candidate outputs.  Any of mbox_* missing means
    # path-3 is blocked regardless of speed — report it.
    tensors = {}
    for name in ("mbox_loc", "mbox_conf_flatten", "mbox_priorbox",
                 "detection_output"):
        try:
            t = interp.getSessionOutput(session, name)
            tensors[name] = t
        except Exception as e:
            tensors[name] = None

    batch_data = _make_batch(batch_size)

    # Warmup
    for _ in range(_WARMUP):
        tmp = MNN.Tensor(
            (batch_size, 1, _H, _W),
            MNN.Halide_Type_Float,
            batch_data,
            MNN.Tensor_DimensionType_Caffe,
        )
        inp.copyFrom(tmp)
        interp.runSession(session)

    # After at least one real run, the dynamic tensor shapes are
    # settled — record them now so we know what shape to expect in
    # the real ``detect_batch`` implementation.
    raw_shapes = {}
    for name, t in tensors.items():
        if t is None:
            raw_shapes[name] = None
            continue
        try:
            raw_shapes[name] = list(t.getShape())
        except Exception as e:
            raw_shapes[name] = f"shape-error: {e}"

    # Timed loop.
    per_call_ms = []
    for _ in range(_ITERS):
        tmp = MNN.Tensor(
            (batch_size, 1, _H, _W),
            MNN.Halide_Type_Float,
            batch_data,
            MNN.Tensor_DimensionType_Caffe,
        )
        inp.copyFrom(tmp)
        t0 = time.perf_counter()
        interp.runSession(session)
        # Force the raw outputs to be materialised on host — this is
        # what the real detect_batch code path will do.
        for name in ("mbox_loc", "mbox_conf_flatten", "mbox_priorbox"):
            t = tensors[name]
            if t is None:
                continue
            shp = t.getShape()
            total = int(np.prod(shp)) if shp and all(s > 0 for s in shp) else 0
            if total == 0:
                continue
            host = MNN.Tensor(
                shp, MNN.Halide_Type_Float,
                np.zeros(shp, dtype=np.float32),
                MNN.Tensor_DimensionType_Caffe,
            )
            t.copyToHostTensor(host)
        t1 = time.perf_counter()
        per_call_ms.append((t1 - t0) * 1000.0)

    # Also record the detection_output numel so we can cross-check
    # that batch>1 still yields 0 (confirms the ``probe_v2`` finding
    # holds inside the same container).
    det_shape = raw_shapes.get("detection_output")
    det_numel = (int(np.prod(det_shape)) if isinstance(det_shape, list)
                 and all(s >= 0 for s in det_shape) else None)

    per_frame_ms = [x / batch_size for x in per_call_ms]
    return {
        "batch_size": batch_size,
        "backend": backend,
        "iters": _ITERS,
        "raw_tensor_shapes": raw_shapes,
        "detection_output_numel": det_numel,
        "per_call_ms_p50": round(statistics.median(per_call_ms), 3),
        "per_call_ms_mean": round(statistics.mean(per_call_ms), 3),
        "per_frame_ms_p50": round(statistics.median(per_frame_ms), 3),
        "per_frame_ms_mean": round(statistics.mean(per_frame_ms), 3),
        "per_call_ms_all": [round(x, 3) for x in per_call_ms],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--backend", default="CPU")
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    args = ap.parse_args()

    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 2

    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]

    report = {
        "mnn_version": MNN.version(),
        "model_path": str(args.model),
        "backend": args.backend,
        "input_fixed_hw": [_H, _W],
        "warmup": _WARMUP,
        "iters_per_batch": _ITERS,
        "probes": [],
    }

    for bs in batch_sizes:
        try:
            entry = _probe_batch(args.model, args.backend, bs)
        except Exception as e:
            entry = {
                "batch_size": bs,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        report["probes"].append(entry)

    # Compute speedup vs batch=1 (per-frame).
    base = next(
        (p for p in report["probes"]
         if p.get("batch_size") == 1 and "per_frame_ms_mean" in p),
        None,
    )
    if base is not None:
        base_pf = base["per_frame_ms_mean"]
        for p in report["probes"]:
            if "per_frame_ms_mean" not in p:
                continue
            p["speedup_vs_batch1"] = round(base_pf / p["per_frame_ms_mean"], 3)

    # Human-readable summary (also goes to stderr).
    print(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n── SUMMARY ──", file=sys.stderr)
    print(f"backend: {args.backend}", file=sys.stderr)
    for p in report["probes"]:
        bs = p.get("batch_size", "?")
        if "error" in p:
            print(f"  batch={bs}: ERROR {p['error']}", file=sys.stderr)
            continue
        pf = p.get("per_frame_ms_mean", "?")
        su = p.get("speedup_vs_batch1", "—")
        det = p.get("detection_output_numel", "?")
        print(
            f"  batch={bs}: per-frame {pf:>6} ms  "
            f"speedup {su}x  det_out_numel={det}",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
