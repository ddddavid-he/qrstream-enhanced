"""Probe MNN detect.mnn output layout with batch_size > 1.

Purpose
-------
Before committing to a real ``detect_batch`` implementation for
``MNNQrDetector`` (Milestone 3, task 3.1), we must confirm how
MNN's ``DetectionOutput`` operator lays out its output when the
input is a batch of N frames.

Known facts (from M0):
- With batch=1, MNN outputs shape ``(1, 1, N_detections, 6)`` where
  the 6 fields are ``[class, conf, x0, y0, x1, y1]`` — **no batch_idx**.
- OpenCV DNN outputs shape ``(1, 1, N_detections, 7)`` with an extra
  ``batch_idx`` field in position 0.

Open question: when we pass batch_size=4 through MNN, does the output
include some form of batch discriminator?  There are three plausible
layouts:

  (a) ``(1, 1, N_total, 6)`` — all detections merged, **no way** to
      tell which frame a detection came from.  This would block a
      real batch implementation without a model-level change.
  (b) ``(1, 1, N_total, 7)`` — MNN re-adds a batch_idx column when
      batch > 1.  We can route detections back to their source frame.
  (c) ``(B, 1, N_per_frame, 6)`` — per-batch-item independent slices,
      same dim as batch=1.
  (d) something else entirely.

This probe runs the real ``detect.mnn`` model with batches of size
1, 2, and 4 on synthetic inputs, records the output shape and the
raw first-row contents for each batch, and writes a JSON report that
the Milestone 3.1 design doc can reference directly.

Running
-------
Designed to run inside a podman container built from
``Containerfile.m3_probe``.  See that file for the exact command.
Standalone invocation::

    python dev/wechatqrcode-mnn-poc/scripts/probe_mnn_batch_output.py \
        --model dev/wechatqrcode-mnn-poc/models/mnn/detect.mnn \
        --output dev/wechatqrcode-mnn-poc/results/mnn_batch_probe.json
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
    print("ERROR: MNN not installed.  Install with: pip install MNN", file=sys.stderr)
    sys.exit(2)


# Use a fixed input size so batch behaviour is the only variable.
# 384×384 matches the upstream SSD default; that way we don't
# conflate dynamic-resize effects with batch effects.
_H, _W = 384, 384


def _make_synthetic_frame(seed: int) -> np.ndarray:
    """Build a deterministic grayscale frame with a fake QR-like pattern.

    We're not trying to actually detect anything — we just need
    the DetectionOutput layer to produce *some* output so we can
    inspect its shape.  A high-contrast checkerboard in the centre
    gives the SSD head a non-zero prior, which usually yields at
    least a few low-confidence detections.
    """
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, size=(_H, _W), dtype=np.uint8).astype(np.float32)

    # Insert a checkerboard in the centre to trip SSD priors.
    cy, cx = _H // 2, _W // 2
    size = 80
    y0, y1 = cy - size, cy + size
    x0, x1 = cx - size, cx + size
    for i in range(y0, y1, 8):
        for j in range(x0, x1, 8):
            val = 0 if ((i // 8) + (j // 8)) % 2 == 0 else 255
            frame[i:i + 8, j:j + 8] = val

    return frame / 255.0


def _probe_single_batch(
    interp: "MNN.Interpreter",
    session,
    batch_size: int,
) -> dict:
    """Run one inference with the given batch_size; return a report entry."""
    # Build the input batch (batch_size, 1, H, W).
    frames = np.stack(
        [_make_synthetic_frame(seed=100 + i) for i in range(batch_size)]
    )
    input_data = frames.reshape(batch_size, 1, _H, _W).astype(np.float32)

    # Resize the session input tensor to the requested batch.
    inp = interp.getSessionInput(session)
    interp.resizeTensor(inp, (batch_size, 1, _H, _W))
    interp.resizeSession(session)

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

    # Copy to host so we can introspect.
    tmp_out = MNN.Tensor(
        out_shape,
        MNN.Halide_Type_Float,
        np.zeros(out_shape, dtype=np.float32),
        MNN.Tensor_DimensionType_Caffe,
    )
    out.copyToHostTensor(tmp_out)
    flat = np.array(tmp_out.getData(), dtype=np.float32)

    # Summarise: up to first 8 "rows" of whatever the inner-most
    # dimension turns out to be.  We try dim=6 and dim=7 separately
    # and see which parsing gives structured output.
    total = int(np.prod(out_shape))
    head = flat[:min(total, 64)].tolist()

    parses = {}
    for dim_candidate in (6, 7):
        if total % dim_candidate == 0:
            rows = total // dim_candidate
            arr = flat[:rows * dim_candidate].reshape(rows, dim_candidate)
            # Only keep rows whose second/third field looks like a
            # plausible (class, confidence) pair — filters out the
            # trailing zeros MNN pads the output with.
            plausible = []
            for r in arr[:min(rows, 16)]:
                row = [round(float(v), 4) for v in r.tolist()]
                plausible.append(row)
            parses[f"dim={dim_candidate}"] = {
                "implied_rows": rows,
                "first_rows": plausible,
            }

    return {
        "batch_size": batch_size,
        "input_shape": [batch_size, 1, _H, _W],
        "output_shape": list(out_shape),
        "output_numel": total,
        "output_head_flat": [round(float(v), 4) for v in head],
        "parse_attempts": parses,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", required=True, type=Path,
        help="Path to detect.mnn",
    )
    ap.add_argument(
        "--output", required=True, type=Path,
        help="Output JSON path for the report",
    )
    ap.add_argument(
        "--backend", default="CPU",
        help="MNN backend (default: CPU)",
    )
    ap.add_argument(
        "--batch-sizes", default="1,2,4,8",
        help="Comma-separated batch sizes to probe",
    )
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
        "probes": [],
    }

    interp = MNN.Interpreter(str(args.model))
    session = interp.createSession({"backend": args.backend})

    for bs in batch_sizes:
        try:
            entry = _probe_single_batch(interp, session, bs)
        except Exception as e:
            entry = {
                "batch_size": bs,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        report["probes"].append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
