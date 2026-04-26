"""
Adaptive sample-rate probe for the decode pipeline.

Measures QR detection rate and repeat count across multiple video
windows to determine the optimal ``sample_rate`` for the main scan.

Split from ``decoder.py`` for readability; all public symbols are
re-exported by ``decoder.py`` so external imports are unchanged.
"""

from __future__ import annotations

from functools import partial
from math import log
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from tqdm import tqdm

from ._video_io import (
    _read_frame_ranges,
    _worker_detect_qr,
    _merge_ranges,
)


def _build_probe_ranges(total_frames: int, window_size: int = 120,
                        gap_ratio: float = 0.15):
    """Build three fixed-size probe windows spread across the middle.

    The windows are centered around 50% of the timeline with a configurable
    percentage gap to the left and right so probe sampling avoids the start/end
    idle regions while still observing separated playback segments.
    """
    if total_frames <= 0 or window_size <= 0:
        return []
    if total_frames <= window_size:
        return [(0, total_frames - 1)]

    half = window_size // 2
    centers = [0.5 - gap_ratio, 0.5, 0.5 + gap_ratio]
    ranges = []
    for ratio in centers:
        ratio = min(max(ratio, 0.0), 1.0)
        center = int(round((total_frames - 1) * ratio))
        start = max(0, center - half)
        end = start + window_size - 1
        if end >= total_frames:
            end = total_frames - 1
            start = max(0, end - window_size + 1)
        ranges.append((start, end))

    return _merge_ranges(ranges)


def _compute_auto_sample_rate(detect_rate: float, avg_repeat: float) -> int:
    """Compute a conservative sample rate from one probe window."""
    TARGET_DETECT_PROB = 0.95
    p = detect_rate

    if p >= 0.99:
        return max(1, int(avg_repeat / 1.5))
    if p > 0.01:
        min_chances = log(1 - TARGET_DETECT_PROB) / log(1 - p)
        return max(1, int(avg_repeat / min_chances))
    return 1


def _analyze_probe_window(window_results):
    """Analyze one contiguous probe window independently."""
    frame_count = len(window_results)
    if frame_count == 0:
        return {
            'frame_count': 0,
            'detect_rate': 0.0,
            'avg_repeat': 1.0,
            'distinct_seed_count': 0,
            'sample_rate': None,
        }

    detected = sum(1 for _, block_bytes, seed in window_results if seed is not None)
    detect_rate = detected / frame_count
    distinct_seeds = {seed for _, _, seed in window_results if seed is not None}

    seed_runs = []
    current_seed = None
    current_run = 0
    for _, _, seed in window_results:
        if seed is not None:
            if seed == current_seed:
                current_run += 1
            else:
                if current_run > 0:
                    seed_runs.append(current_run)
                current_seed = seed
                current_run = 1
    if current_run > 0:
        seed_runs.append(current_run)

    avg_repeat = sum(seed_runs) / len(seed_runs) if seed_runs else 1.0
    sample_rate = None
    if len(distinct_seeds) >= 2:
        sample_rate = _compute_auto_sample_rate(detect_rate, avg_repeat)

    return {
        'frame_count': frame_count,
        'detect_rate': detect_rate,
        'avg_repeat': avg_repeat,
        'distinct_seed_count': len(distinct_seeds),
        'sample_rate': sample_rate,
    }


def _probe_sample_rate(video_path: str, workers: int,
                       verbose: bool = False, qr_detector=None):
    """Probe multiple windows of a video to determine optimal sample_rate.

    Returns:
        (sample_rate, probe_results, probe_count, leading_frames_probed,
         detect_rate, avg_repeat)
    """
    PROBE_WINDOW_SIZE = 120
    PROBE_GAP_RATIO = 0.15

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    probe_ranges = _build_probe_ranges(total_frames, PROBE_WINDOW_SIZE, PROBE_GAP_RATIO)
    probe_frames = list(_read_frame_ranges(video_path, probe_ranges))
    probe_count = len(probe_frames)
    leading_frames_probed = 0

    if not probe_frames:
        return 1, [], 0, 0, 0.0, 1.0

    if verbose and len(probe_ranges) > 1:
        ranges_str = ", ".join(f"{start}-{end}" for start, end in probe_ranges)
        print(f"Probe windows: {ranges_str}")

    worker = partial(_worker_detect_qr, qr_detector=qr_detector)
    probe_results = []
    pbar = tqdm(total=probe_count, desc="Probe",
                unit="f", dynamic_ncols=True,
                mininterval=0, miniters=1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, fd): fd[0]
                   for fd in probe_frames}
        for future in as_completed(futures):
            result = future.result()
            probe_results.append(result)
            pbar.update(1)
    pbar.close()

    # Sort by frame index
    probe_results.sort(key=lambda x: x[0])

    window_stats = []
    for start, end in probe_ranges:
        window_results = [result for result in probe_results if start <= result[0] <= end]
        stats = _analyze_probe_window(window_results)
        window_stats.append((start, end, stats))

    valid_windows = [entry for entry in window_stats if entry[2]['sample_rate'] is not None]
    if not valid_windows:
        print(f"Probe: {probe_count} frames, insufficient seed diversity → sample_rate=1")
        return 1, probe_results, probe_count, leading_frames_probed, 0.0, 1.0

    limiting_start, limiting_end, limiting_stats = min(
        valid_windows,
        key=lambda entry: entry[2]['sample_rate'],
    )
    auto_rate = limiting_stats['sample_rate']
    detect_rate = limiting_stats['detect_rate']
    avg_run = limiting_stats['avg_repeat']

    if verbose:
        for start, end, stats in window_stats:
            rate_str = stats['sample_rate'] if stats['sample_rate'] is not None else 'n/a'
            print(
                f"  Probe window {start}-{end}: detect_rate={stats['detect_rate']:.0%}, "
                f"avg_repeat={stats['avg_repeat']:.1f}, seeds={stats['distinct_seed_count']}, "
                f"sample_rate={rate_str}"
            )

    print(f"Probe: {probe_count} frames across {len(probe_ranges)} windows, "
          f"limiting_window={limiting_start}-{limiting_end}, "
          f"detect_rate={detect_rate:.0%}, avg_repeat={avg_run:.1f} → sample_rate={auto_rate}")

    return (auto_rate, probe_results, probe_count,
            leading_frames_probed, detect_rate, avg_run)
