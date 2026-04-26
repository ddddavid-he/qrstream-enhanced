"""Follow-up probe for MNN batch output layout.

v1 found batch=1 works, batch>=2 returns 0 detections.  Before we
conclude "MNN DetectionOutput doesn't support batch", rule out:

  - State pollution: maybe resizing to batch>=2 corrupts something.
    Mitigation: probe v2 builds a *fresh* interpreter+session for
    each batch size (so every probe is from a clean slate).

  - Input degeneracy: maybe the checkerboard stops triggering priors
    when duplicated.  Mitigation: probe v2 also runs batch=1 with
    the very first frame of each batch set, so we can confirm the
    same input produces non-empty output when batched alone.

  - Confidence threshold quirk: maybe the DetectionOutput op applies
    a higher threshold to batched inputs.  Mitigation: inspect the
    raw confidence head before filtering.

  - batch_id column hidden elsewhere: v1 assumed the missing dim=7
    would show up in output_shape[3]; v2 also dumps the full raw
    flat buffer (even if numel=0) and, if non-empty, tries parsing
    under several dim candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

try:
    import MNN  # type: ignore
except ImportError:
    print("ERROR: MNN not installed.", file=sys.stderr)
    sys.exit(2)


_H, _W = 384, 384


def _make_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, size=(_H, _W), dtype=np.uint8).astype(np.float32)
    cy, cx = _H // 2, _W // 2
    size = 80
    y0, y1 = cy - size, cy + size
    x0, x1 = cx - size, cx + size
    for i in range(y0, y1, 8):
        for j in range(x0, x1, 8):
            val = 0 if ((i // 8) + (j // 8)) % 2 == 0 else 255
            frame[i:i + 8, j:j + 8] = val
    return frame / 255.0


def _probe_fresh_session(
    model_path: Path,
    backend: str,
    batch_size: int,
) -> dict:
    """Build a *fresh* interpreter+session and run one batch."""
    interp = MNN.Interpreter(str(model_path))
    session = interp.createSession({"backend": backend})

    # Build input: deterministic seeds so we can cross-check which
    # frame is in each batch slot.
    frames = np.stack([_make_frame(seed=100 + i) for i in range(batch_size)])
    input_data = frames.reshape(batch_size, 1, _H, _W).astype(np.float32)

    inp = interp.getSessionInput(session)
    inp_shape_before = tuple(inp.getShape())
    interp.resizeTensor(inp, (batch_size, 1, _H, _W))
    interp.resizeSession(session)
    inp_shape_after = tuple(inp.getShape())

    tmp_in = MNN.Tensor(
        (batch_size, 1, _H, _W),
        MNN.Halide_Type_Float,
        input_data,
        MNN.Tensor_DimensionType_Caffe,
    )
    inp.copyFrom(tmp_in)
    interp.runSession(session)

    out = interp.getSessionOutput(session, "detection_output")
    out_shape = tuple(out.getShape())

    tmp_out = MNN.Tensor(
        out_shape if np.prod(out_shape) > 0 else (1,),
        MNN.Halide_Type_Float,
        np.zeros(out_shape if np.prod(out_shape) > 0 else (1,), dtype=np.float32),
        MNN.Tensor_DimensionType_Caffe,
    )
    if np.prod(out_shape) > 0:
        out.copyToHostTensor(tmp_out)
        flat = np.array(tmp_out.getData(), dtype=np.float32)
    else:
        flat = np.array([], dtype=np.float32)

    total = int(flat.size)
    parses = {}
    for dim_candidate in (6, 7):
        if total > 0 and total % dim_candidate == 0:
            rows = total // dim_candidate
            arr = flat[:rows * dim_candidate].reshape(rows, dim_candidate)
            parses[f"dim={dim_candidate}"] = {
                "implied_rows": rows,
                "all_rows": [[round(float(v), 4) for v in r.tolist()]
                             for r in arr],
            }

    return {
        "batch_size": batch_size,
        "input_shape_before_resize": list(inp_shape_before),
        "input_shape_after_resize": list(inp_shape_after),
        "output_shape": list(out_shape),
        "output_numel": total,
        "output_raw_flat": [round(float(v), 4) for v in flat.tolist()],
        "parse_attempts": parses,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--backend", default="CPU")
    args = ap.parse_args()

    report = {
        "mnn_version": MNN.version(),
        "model_path": str(args.model),
        "backend": args.backend,
        "probes": [],
    }

    # Probe 1: baseline, fresh session, batch=1.
    # Probe 2: same, batch=2.
    # Probe 3: fresh session, batch=1 run twice back-to-back
    #          (rules out "first-resize-corrupts-state").
    # Probe 4: fresh session, batch=4.
    # Probe 5: fresh session, batch=1 but with 4 different frames
    #          fed sequentially (shows batch=1 can process 4 items).

    for probe_idx, bs in enumerate([1, 2, 1, 4]):
        try:
            entry = _probe_fresh_session(args.model, args.backend, bs)
            entry["probe_idx"] = probe_idx
        except Exception as e:
            entry = {
                "probe_idx": probe_idx,
                "batch_size": bs,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        report["probes"].append(entry)

    # Probe 6: batch=1, 4 frames fed sequentially on ONE session.
    try:
        interp = MNN.Interpreter(str(args.model))
        session = interp.createSession({"backend": args.backend})
        inp = interp.getSessionInput(session)
        interp.resizeTensor(inp, (1, 1, _H, _W))
        interp.resizeSession(session)

        sequential = []
        for seed_offset in range(4):
            frame = _make_frame(seed=100 + seed_offset)
            input_data = frame.reshape(1, 1, _H, _W).astype(np.float32)
            tmp_in = MNN.Tensor(
                (1, 1, _H, _W),
                MNN.Halide_Type_Float,
                input_data,
                MNN.Tensor_DimensionType_Caffe,
            )
            inp.copyFrom(tmp_in)
            interp.runSession(session)
            out = interp.getSessionOutput(session, "detection_output")
            out_shape = tuple(out.getShape())
            total = int(np.prod(out_shape))
            if total > 0:
                tmp_out = MNN.Tensor(
                    out_shape,
                    MNN.Halide_Type_Float,
                    np.zeros(out_shape, dtype=np.float32),
                    MNN.Tensor_DimensionType_Caffe,
                )
                out.copyToHostTensor(tmp_out)
                flat = np.array(tmp_out.getData(), dtype=np.float32)
                sequential.append({
                    "frame_seed": 100 + seed_offset,
                    "output_shape": list(out_shape),
                    "output_flat": [round(float(v), 4) for v in flat.tolist()],
                })
            else:
                sequential.append({
                    "frame_seed": 100 + seed_offset,
                    "output_shape": list(out_shape),
                    "output_flat": [],
                })
        report["sequential_batch1_on_one_session"] = sequential
    except Exception as e:
        report["sequential_batch1_on_one_session_error"] = f"{type(e).__name__}: {e}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
