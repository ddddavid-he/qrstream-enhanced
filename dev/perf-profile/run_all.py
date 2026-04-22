"""
Run the full profiling suite.

Uses default sizes (1KB, 10KB, 100KB, 1MB, 5MB, 10MB).
Single-process cProfile is capped at 1MB by default because it's slow.

Override via environment or flags on the individual scripts.

Usage:
    python dev/perf-profile/run_all.py
    python dev/perf-profile/run_all.py --sizes 10,100,1024
    python dev/perf-profile/run_all.py --quick         # skip 5MB+10MB
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n{'#' * 72}")
    print(f"# {name}")
    print(f"# $ {' '.join(cmd)}")
    print(f"{'#' * 72}\n", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0
    print(f"\n[{name}] finished in {elapsed:.1f}s (rc={rc})")
    if rc != 0:
        print(f"WARNING: {name} exited with code {rc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default=None,
                    help="Comma-separated sizes in KB, e.g. '1,10,100,1024'")
    ap.add_argument("--quick", action="store_true",
                    help="Skip 5MB and 10MB runs to save time.")
    ap.add_argument("--skip-hotpaths", action="store_true")
    ap.add_argument("--skip-encode", action="store_true")
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--skip-cprofile", action="store_true",
                    help="Disable single-process cProfile for encode/decode "
                    "(only run staged multi-process timing).")
    ap.add_argument("--cprofile-max-kb", type=int, default=1024,
                    help="Max file size (KB) for cProfile runs (single process).")
    args = ap.parse_args()

    python = sys.executable

    # Decide sizes
    if args.sizes:
        size_arg = ["--sizes", args.sizes]
    elif args.quick:
        size_arg = ["--sizes", "1,10,100,1024"]
    else:
        size_arg = []  # use script default (1,10,100,1024,5120,10240 KB)

    common = size_arg + ["--cprofile-max-kb", str(args.cprofile_max_kb)]
    if args.skip_cprofile:
        common.append("--skip-cprofile")

    suite_start = time.perf_counter()

    if not args.skip_hotpaths:
        run_step("hotpaths micro-bench",
                 [python, str(HERE / "profile_hotpaths.py")])

    if not args.skip_encode:
        run_step("encode profile",
                 [python, str(HERE / "profile_encode.py"), *common])

    if not args.skip_decode:
        run_step("decode profile",
                 [python, str(HERE / "profile_decode.py"), *common])

    elapsed = time.perf_counter() - suite_start
    print(f"\n{'=' * 72}")
    print(f"Full suite finished in {elapsed:.1f}s")
    print(f"Results in: {RESULTS}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
