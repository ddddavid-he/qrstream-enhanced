"""Generate larger benchmark vectors for WASM/Python decoder perf tests.

Outputs bench_vectors.json with incompressible random payloads at
several sizes so the benchmark measures pure decode throughput.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from generate_vectors import make_case  # noqa: E402


def main() -> None:
    cases = [
        make_case("bench_warmup", os.urandom(4_000), overhead=1.2),
        make_case("bench_100k", os.urandom(100_000), overhead=1.2),
        make_case("bench_1m", os.urandom(1_000_000), overhead=1.2),
        make_case("bench_5m", os.urandom(5_000_000), overhead=1.2),
    ]

    out_path = os.path.join(os.path.dirname(__file__), "bench_vectors.json")
    with open(out_path, "w") as f:
        json.dump({"cases": cases}, f)
    print(f"Wrote {len(cases)} cases, {os.path.getsize(out_path)} bytes -> {out_path}")
    for c in cases:
        print(f"  {c['name']}: payload={len(__import__('base64').b64decode(c['payload_b64']))} "
              f"K={c['K']} frames={c['num_frames']}")


if __name__ == "__main__":
    main()
