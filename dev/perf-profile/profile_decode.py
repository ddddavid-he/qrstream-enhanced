"""
Detailed profiling of the decode pipeline.

Generates test videos for several file sizes, then profiles the decode path
with cProfile (single-process) and staged timing (multi-process).

Staged timing breaks decode wall time into:
  - frame read + JPEG encode (producer side)
  - QR detect + base45/base64/COBS + unpack (worker pool)
  - LT belief propagation (main thread: LTDecoder.decode_bytes)
  - probe phase (separate)

Usage:
    python dev/perf-profile/profile_decode.py
    python dev/perf-profile/profile_decode.py --sizes 10,100,1000
    python dev/perf-profile/profile_decode.py --skip-cprofile
"""

import argparse
import cProfile
import io
import os
import pstats
import struct
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from qrstream.encoder import encode_to_video  # noqa: E402
from qrstream.decoder import (  # noqa: E402
    LTDecoder,
    _downscale_frame,
    _worker_detect_qr,
    extract_qr_from_video,
    _probe_sample_rate,
)


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Cache for encoded test videos, keyed by (size_bytes).
_VIDEO_CACHE: dict[int, str] = {}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_temp_input(size_bytes: int) -> str:
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


def _prepare_video(size_bytes: int, workers: int) -> str:
    """Encode a temp input of the given size to an .mp4 video.

    Caches per-process to avoid re-encoding the same size.
    """
    if size_bytes in _VIDEO_CACHE and os.path.exists(_VIDEO_CACHE[size_bytes]):
        return _VIDEO_CACHE[size_bytes]

    input_path = _make_temp_input(size_bytes)
    video_path = tempfile.mktemp(suffix=".mp4")
    try:
        encode_to_video(input_path, video_path,
                        overhead=2.0, fps=10,
                        workers=workers, verbose=False)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    _VIDEO_CACHE[size_bytes] = video_path
    return video_path


def _cleanup_videos() -> None:
    for path in _VIDEO_CACHE.values():
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
    _VIDEO_CACHE.clear()


# ─────────────────────────────────────────────────────────────
# cProfile single-process decode
# ─────────────────────────────────────────────────────────────

def profile_decode_single_process(video_path: str, label: str) -> str:
    """Run extract_qr_from_video with workers=1 under cProfile."""
    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.perf_counter()
    blocks = extract_qr_from_video(video_path, sample_rate=0,
                                    verbose=False, workers=1)
    elapsed = time.perf_counter() - t0
    profiler.disable()

    prof_path = RESULTS_DIR / f"decode_single_{label}.prof"
    profiler.dump_stats(str(prof_path))

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.strip_dirs().sort_stats("tottime").print_stats(25)

    header = (
        f"\n{'=' * 70}\n"
        f" extract_qr_from_video  size={label}  workers=1  "
        f"wall={elapsed:.3f}s  blocks={len(blocks)}\n"
        f"{'=' * 70}\n"
        f"Saved: {prof_path.name}\n"
        f"Top 25 functions by tottime:\n\n"
    )
    return header + buf.getvalue()


# ─────────────────────────────────────────────────────────────
# Staged multi-process timing for decode
#
# We can't easily hook into the main extract_qr_from_video without
# refactoring, so we re-implement a minimal decoder pipeline with
# instrumentation.  This measures the same substages as production.
# ─────────────────────────────────────────────────────────────

def staged_decode_timing(video_path: str, workers: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ── Probe phase ──
    t_probe = time.perf_counter()
    (sample_rate, probe_results, probe_count,
     leading_probed, detect_rate, avg_repeat) = _probe_sample_rate(
        video_path, workers=workers, verbose=False)
    time_probe = time.perf_counter() - t_probe

    # Feed probe results into LT decoder
    lt_decoder = LTDecoder()
    seen_seeds: set[int] = set()
    unique_blocks: list[bytes] = []
    time_lt = 0.0

    for fidx, block_bytes, seed in probe_results:
        if block_bytes is not None and seed is not None:
            if seed not in seen_seeds:
                seen_seeds.add(seed)
                unique_blocks.append(block_bytes)
                t0 = time.perf_counter()
                try:
                    done, _ = lt_decoder.decode_bytes(block_bytes, skip_crc=True)
                    time_lt += time.perf_counter() - t0
                    if done:
                        return {
                            "probe": time_probe,
                            "frame_read": 0.0,
                            "worker_wait": 0.0,
                            "lt_decode": time_lt,
                            "wall": time.perf_counter() - t_probe,
                            "sample_rate": sample_rate,
                            "blocks": len(unique_blocks),
                            "K": lt_decoder.K,
                            "early": "during probe",
                            "workers": workers,
                        }
                except (ValueError, struct.error):
                    pass

    # ── Main scan ──
    BATCH_SIZE = workers * 4
    time_frame_read = 0.0   # frame read + downscale + jpeg encode (producer)
    time_worker_wait = 0.0  # waiting on pool results (includes IPC)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    t_scan_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        batch: list = []

        # Reimplement _read_frames with timing.
        def _iter_frames():
            nonlocal time_frame_read
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            while True:
                t_read = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    time_frame_read += time.perf_counter() - t_read
                    break
                if frame_idx % sample_rate == 0:
                    frame = _downscale_frame(frame)
                    _, jpeg_bytes = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    time_frame_read += time.perf_counter() - t_read
                    yield (frame_idx, jpeg_bytes.tobytes())
                else:
                    time_frame_read += time.perf_counter() - t_read
                frame_idx += 1
            cap.release()

        early_done = False

        def _process_batch(batch_items):
            nonlocal time_worker_wait, time_lt, early_done
            t_w = time.perf_counter()
            futures = [executor.submit(_worker_detect_qr, fd) for fd in batch_items]
            results = [f.result() for f in futures]
            time_worker_wait += time.perf_counter() - t_w

            for fidx, block_bytes, seed in results:
                if block_bytes is None or seed is None:
                    continue
                if seed in seen_seeds:
                    continue
                seen_seeds.add(seed)
                unique_blocks.append(block_bytes)
                t_lt = time.perf_counter()
                try:
                    done, _ = lt_decoder.decode_bytes(block_bytes, skip_crc=True)
                    time_lt += time.perf_counter() - t_lt
                    if done:
                        early_done = True
                        return
                except (ValueError, struct.error):
                    time_lt += time.perf_counter() - t_lt

        for frame_data in _iter_frames():
            batch.append(frame_data)
            if len(batch) >= BATCH_SIZE:
                _process_batch(batch)
                batch.clear()
                if early_done:
                    break
        if batch and not early_done:
            _process_batch(batch)

    total_wall = time.perf_counter() - t_probe

    # NOTE: time_frame_read and time_worker_wait partially overlap in real
    # execution (producer yields while pool runs), but our serialised
    # measurement charges them separately.  We report raw numbers.

    return {
        "probe": time_probe,
        "frame_read": time_frame_read,
        "worker_wait": time_worker_wait,
        "lt_decode": time_lt,
        "wall": total_wall,
        "sample_rate": sample_rate,
        "blocks": len(unique_blocks),
        "K": lt_decoder.K if lt_decoder.initialized else 0,
        "early": "yes" if (lt_decoder.initialized and lt_decoder.done) else "no",
        "workers": workers,
        "total_frames": total_frames,
    }


def format_staged_report(label: str, stats: dict) -> str:
    wall = stats["wall"]
    lines = [
        f"\n{'=' * 70}",
        f" staged decode  size={label}  workers={stats['workers']}",
        f" wall={wall:.3f}s  sample_rate={stats['sample_rate']}  "
        f"blocks={stats['blocks']}  K={stats['K']}  early={stats['early']}",
        f"{'=' * 70}",
        f"  probe phase               : {stats['probe']:8.3f}s  "
        f"({_format_pct(stats['probe'], wall)})",
        f"  frame read + jpeg encode  : {stats['frame_read']:8.3f}s  "
        f"({_format_pct(stats['frame_read'], wall)})",
        f"  worker wait (detect+IPC)  : {stats['worker_wait']:8.3f}s  "
        f"({_format_pct(stats['worker_wait'], wall)})",
        f"  LT belief propagation     : {stats['lt_decode']:8.3f}s  "
        f"({_format_pct(stats['lt_decode'], wall)})",
    ]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

DEFAULT_SIZES_KB = [1, 10, 100, 1024, 5 * 1024, 10 * 1024]


def parse_sizes(spec: str) -> list[int]:
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
            out.append(int(tok) * 1024)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default=None)
    ap.add_argument("--skip-cprofile", action="store_true")
    ap.add_argument("--skip-staged", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cprofile-max-kb", type=int, default=1024,
                    help="Cap cProfile runs to videos produced from "
                    "files <= this size (KB).")
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes) if args.sizes else [s * 1024 for s in DEFAULT_SIZES_KB]
    workers = args.workers or (os.cpu_count() or 4)

    report_path = RESULTS_DIR / "decode_report.txt"
    report_lines: list[str] = []

    try:
        for size_bytes in sizes:
            label = _human_size(size_bytes)
            print(f"\n>>> Preparing video for {label} ...", flush=True)
            video_path = _prepare_video(size_bytes, workers=workers)
            vid_size = os.path.getsize(video_path)
            intro = f"[{label}] video={vid_size / 1024 / 1024:.1f}MB\n"
            print(intro, flush=True)
            report_lines.append(intro)

            # Staged timing
            if not args.skip_staged:
                print(f">>> staged decode  size={label}  workers={workers}",
                      flush=True)
                stats = staged_decode_timing(video_path, workers=workers)
                section = format_staged_report(label, stats)
                print(section)
                report_lines.append(section)

            # cProfile single-process
            if not args.skip_cprofile and size_bytes <= args.cprofile_max_kb * 1024:
                print(f">>> cProfile decode  size={label}  workers=1",
                      flush=True)
                section = profile_decode_single_process(video_path, label)
                print(section)
                report_lines.append(section)
            elif not args.skip_cprofile:
                skip_note = (
                    f"\n(Skipping cProfile for {label}: exceeds "
                    f"--cprofile-max-kb={args.cprofile_max_kb} KB)\n"
                )
                print(skip_note)
                report_lines.append(skip_note)
    finally:
        _cleanup_videos()

    report_path.write_text("".join(report_lines))
    print(f"\nFull report written to: {report_path}")


if __name__ == "__main__":
    main()
