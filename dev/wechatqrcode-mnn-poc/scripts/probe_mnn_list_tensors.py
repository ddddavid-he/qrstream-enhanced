"""List every input/output tensor MNN knows about in detect.mnn.

The perf probe couldn't find ``mbox_loc`` / ``mbox_conf_flatten`` /
``mbox_priorbox`` — MNN's Caffe importer or offline optimiser may have
renamed / fused them.  We need to know the real tensor names before
the path-3 implementation can ask for them via ``getSessionOutput``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import MNN  # type: ignore
except ImportError:
    print("ERROR: MNN not installed.", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    interp = MNN.Interpreter(str(args.model))
    session = interp.createSession({"backend": "CPU"})

    # Force the interpreter to materialise input shape before we ask
    # for tensor metadata, so getShape() reports real dimensions.
    inp = interp.getSessionInput(session)
    interp.resizeTensor(inp, (1, 1, 384, 384))
    interp.resizeSession(session)

    # Run once so any deferred tensors have their shapes computed.
    dummy = np.zeros((1, 1, 384, 384), dtype=np.float32)
    tmp = MNN.Tensor(
        (1, 1, 384, 384), MNN.Halide_Type_Float, dummy,
        MNN.Tensor_DimensionType_Caffe,
    )
    inp.copyFrom(tmp)
    interp.runSession(session)

    # getSessionInputAll / getSessionOutputAll return dicts {name: tensor}.
    inputs = interp.getSessionInputAll(session) or {}
    outputs = interp.getSessionOutputAll(session) or {}

    def _describe(d: dict) -> dict:
        out = {}
        for name, t in d.items():
            try:
                shp = list(t.getShape())
            except Exception as e:
                shp = f"shape-error: {e}"
            out[name] = shp
        return out

    report = {
        "mnn_version": MNN.version(),
        "model_path": str(args.model),
        "inputs": _describe(inputs),
        "outputs": _describe(outputs),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
