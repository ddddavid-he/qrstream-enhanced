"""
Detailed profiling of the encode pipeline.

Two complementary measurements:

1. Single-process cProfile — function-level hotspots (accurate, but slower
   than production because workers=1).
2. Multi-process staged timing — wall-time broken down into:
   - LT block generation (producer thread)
   - QR image generation (worker pool)
   - VideoWriter.write (main thread)
   - IPC / scheduling overhead (difference)

Usage:
    python dev/perf-profile/profile_encode.py
    python dev/perf-profile/profile_encode.py --sizes 10,100,1000
    python dev/perf-profile/profile_encode.py --skip-cprofile
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
import tempfile
import time
from itertools import repeat
from math import ceil
from pathlib import Path

# Add src/ to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from qrstream.encoder import LTEncoder, encode_to_video, _load_payload  # noqa: E402
from qrstream.protocol import auto_blocksize  # noqa: E402
from qrstream.qr_utils import generate_qr_image  # noqa: E402


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_temp_input(size_bytes: int) -> str:
    """Create a temp file of the given size filled with random data."""
    fd, path = tempfile.mkstemp(suffix=".bin")
    with os.fdopen(fd, "wb") as f:
        f.write(os.urandom(size_bytes))
    return path


def _human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes // 1024}KB"
    return f"{size_bytes // (1024 * 1024)}MB"


def _format_pct(part: float, whole: float) -> str:
    if whole <= 0:
        return "  n/a"
    return f"{part / whole * 100:5.1f}%"


# ─────────────────────────────────────────────────────────────
# cProfile single-process run
# ─────────────────────────────────────────────────────────────

def profile_encode_single_process(input_path: str, output_path: str,
                                  label: str) -> str:
    """Run encode_to_video with workers=1 under cProfile.

    Returns a formatted text report and writes the .prof file.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.perf_counter()
    encode_to_video(
        input_path,
        output_path,
        overhead=2.0,
        fps=10,
        workers=1,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    profiler.disable()

    # Save raw .prof for snakeviz
    prof_path = RESULTS_DIR / f"encode_single_{label}.prof"
    profiler.dump_stats(str(prof_path))

    # Build human-readable report
    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.strip_dirs().sort_stats("tottime").print_stats(25)

    header = (
        f"\n{'=' * 70}\n"
        f" encode_to_video  size={label}  workers=1  wall={elapsed:.3f}s\n"
        f"{'=' * 70}\n"
        f"Saved: {prof_path.name}\n"
        f"Top 25 functions by tottime:\n\n"
    )
    return header + buf.getvalue()


# ─────────────────────────────────────────────────────────────
# Multi-process staged timing (no cProfile, measures real wall time)
# ─────────────────────────────────────────────────────────────

def staged_encode_timing(input_path: str, workers: int) -> dict:
    """Re-implement encode_to_video with fine-grained timing.

    Mirrors the real pipeline but records time spent in each stage.
    Does NOT write a video to disk (VideoWriter.write is timed separately
    to an in-memory sink via a fake path we then discard) — but OpenCV's
    VideoWriter does not support in-memory sinks portably, so we still
    write to /tmp and then delete.
    """
    from queue import Queue
    from threading import Thread
    from concurrent.futures import ThreadPoolExecutor

    # Setup (mimicking encode_to_video)
    payload, compress, used_mmap, raw_size = _load_payload(
        input_path, compress=True,
        force_compress=False, verbose=False,
    )
    payload_size = len(payload)

    # v0.6.0 defaults: V25, base45 alphanumeric QR
    ec_level, qr_version, alphanumeric_qr = 1, 25, True
    overhead, fps = 2.0, 10
    blocksize = auto_blocksize(payload_size, ec_level, qr_version,
                               alphanumeric_qr=alphanumeric_qr)
    K = ceil(payload_size / blocksize)
    num_blocks = int(K * overhead)

    encoder = LTEncoder(payload, blocksize, compressed=compress,
                        alphanumeric_qr=alphanumeric_qr)

    # Size probe
    first_packed, _, _ = next(encoder.generate_blocks(1))
    first_qr = generate_qr_image(first_packed, ec_level=ec_level,
                                 box_size=10, border=4.0,
                                 version=qr_version, use_legacy=False,
                                 alphanumeric=alphanumeric_qr)
    h, w = first_qr.shape[:2]

    output_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Timing accumulators
    time_producer = 0.0  # LT block generation (in producer thread)
    time_qr_pool = 0.0   # pool.map wait (QR generation + IPC)
    time_write = 0.0     # writer.write
    time_resize = 0.0    # cv2.resize if shape mismatch
    total_start = time.perf_counter()

    batch_size = max(workers * 4, 64)
    block_queue: Queue = Queue(maxsize=batch_size * 2)

    def _block_producer():
        nonlocal time_producer
        encoder._seq = 0
        for packed, _, _ in encoder.generate_blocks(num_blocks):
            t0 = time.perf_counter()
            block_queue.put(packed)
            # time spent blocking on put is considered back-pressure, not
            # generation; we count generation only via the producer's
            # own cProfile-style measure by putting a timestamp before.
            time_producer += time.perf_counter() - t0
        block_queue.put(None)

    producer = Thread(target=_block_producer, daemon=True)
    producer.start()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = False
        while not done:
            batch = []
            for _ in range(batch_size):
                item = block_queue.get()
                if item is None:
                    done = True
                    break
                batch.append(item)
            if not batch:
                break

            t_pool = time.perf_counter()
            # Positional args order of generate_qr_image:
            #   (data, ec_level, box_size, border, version,
            #    use_legacy, binary_mode, alphanumeric)
            qr_imgs = list(pool.map(
                generate_qr_image, batch,
                repeat(ec_level), repeat(10), repeat(4.0),
                repeat(qr_version), repeat(False),
                repeat(None), repeat(alphanumeric_qr),
            ))
            time_qr_pool += time.perf_counter() - t_pool

            for qr_img in qr_imgs:
                if qr_img.shape[:2] != (h, w):
                    t_r = time.perf_counter()
                    qr_img = cv2.resize(qr_img, (w, h),
                                        interpolation=cv2.INTER_NEAREST)
                    time_resize += time.perf_counter() - t_r
                t_w = time.perf_counter()
                writer.write(qr_img)
                time_write += time.perf_counter() - t_w

    producer.join(timeout=5)
    writer.release()
    total_wall = time.perf_counter() - total_start

    # Cleanup
    try:
        os.remove(output_path)
    except OSError:
        pass
    close = getattr(payload, "close", None)
    if callable(close):
        close()

    # "Other" includes scheduling, thread overhead, mmap setup, etc.
    accounted = time_qr_pool + time_write + time_resize
    other = max(0.0, total_wall - accounted)

    return {
        "wall": total_wall,
        "num_blocks": num_blocks,
        "K": K,
        "blocksize": blocksize,
        "frame_size": (w, h),
        "qr_pool": time_qr_pool,
        "writer_write": time_write,
        "resize": time_resize,
        "producer_put_wait": time_producer,
        "other": other,
        "workers": workers,
    }


def format_staged_report(label: str, stats: dict) -> str:
    wall = stats["wall"]
    lines = [
        f"\n{'=' * 70}",
        f" staged encode  size={label}  workers={stats['workers']}",
        f" wall={wall:.3f}s  frames={stats['num_blocks']}  "
        f"K={stats['K']}  blocksize={stats['blocksize']}",
        f" frame={stats['frame_size'][0]}x{stats['frame_size'][1]}",
        f"{'=' * 70}",
        f"  pool.map (QR gen + IPC)   : {stats['qr_pool']:8.3f}s  "
        f"({_format_pct(stats['qr_pool'], wall)})",
        f"  VideoWriter.write         : {stats['writer_write']:8.3f}s  "
        f"({_format_pct(stats['writer_write'], wall)})",
        f"  cv2.resize (if mismatch)  : {stats['resize']:8.3f}s  "
        f"({_format_pct(stats['resize'], wall)})",
        f"  producer put() backpress. : {stats['producer_put_wait']:8.3f}s  "
        f"(overlapped w/ pool)",
        f"  other (sched/startup/etc) : {stats['other']:8.3f}s  "
        f"({_format_pct(stats['other'], wall)})",
        f"  throughput                : {stats['num_blocks'] / wall:6.1f} frames/s",
    ]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

DEFAULT_SIZES_KB = [1, 10, 100, 1024, 5 * 1024, 10 * 1024]


def parse_sizes(spec: str) -> list[int]:
    """Parse '--sizes 1,10,100' into [1024, 10240, 102400]."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower().endswith("mb"):
            out.append(int(tok[:-2]) * 1024 * 1024)
        elif tok.lower().endswith("kb"):
            out.append(int(tok[:-2]) * 1024)
        else:
            out.append(int(tok) * 1024)  # default: KB
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default=None,
                    help="Comma-separated list in KB, e.g. '1,10,100,1024'")
    ap.add_argument("--skip-cprofile", action="store_true",
                    help="Skip the slow single-process cProfile run.")
    ap.add_argument("--skip-staged", action="store_true",
                    help="Skip the multi-process staged timing.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Worker count for staged timing. Default: os.cpu_count().")
    ap.add_argument("--cprofile-max-kb", type=int, default=1024,
                    help="Cap cProfile runs to files <= this size (KB). "
                    "Single-process is slow; default 1024 = 1MB.")
    args = ap.parse_args()

    if args.sizes:
        sizes = parse_sizes(args.sizes)
    else:
        sizes = [s * 1024 for s in DEFAULT_SIZES_KB]

    workers = args.workers or (os.cpu_count() or 4)

    report_path = RESULTS_DIR / "encode_report.txt"
    report_lines: list[str] = []

    for size_bytes in sizes:
        label = _human_size(size_bytes)
        print(f"\n>>> Preparing {label} input ...", flush=True)
        input_path = _make_temp_input(size_bytes)

        try:
            # Staged multi-process timing
            if not args.skip_staged:
                print(f">>> staged encode  size={label}  workers={workers}",
                      flush=True)
                stats = staged_encode_timing(input_path, workers=workers)
                section = format_staged_report(label, stats)
                print(section)
                report_lines.append(section)

            # Single-process cProfile (slower; skip for very large inputs)
            if not args.skip_cprofile and size_bytes <= args.cprofile_max_kb * 1024:
                print(f">>> cProfile encode  size={label}  workers=1",
                      flush=True)
                output_path = tempfile.mktemp(suffix=".mp4")
                try:
                    section = profile_encode_single_process(
                        input_path, output_path, label)
                    print(section)
                    report_lines.append(section)
                finally:
                    if os.path.exists(output_path):
                        os.remove(output_path)
            elif not args.skip_cprofile:
                skip_note = (
                    f"\n(Skipping cProfile for {label}: exceeds "
                    f"--cprofile-max-kb={args.cprofile_max_kb} KB)\n"
                )
                print(skip_note)
                report_lines.append(skip_note)
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    report_path.write_text("".join(report_lines))
    print(f"\nFull report written to: {report_path}")


if __name__ == "__main__":
    main()
