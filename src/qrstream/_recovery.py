"""
Targeted recovery for missing LT seeds.

After the main scan fails to recover all LT source blocks, this
module estimates the video frame positions of missing seeds using
a linear regression model, then rescans those segments with CLAHE
enhancement.

Split from ``decoder.py`` for readability; all public symbols are
re-exported by ``decoder.py`` so external imports are unchanged.
"""

from __future__ import annotations

from functools import partial
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from ._video_io import (
    _read_frame_ranges,
    _worker_detect_qr_clahe,
    _merge_ranges,
    _stream_scan,
)


def _estimate_frame_for_seed(seed: int, seed_frame_map: dict[int, int],
                             frames_per_qr: float,
                             total_frames: int) -> int:
    """Estimate the video frame where a given seed is likely located.

    Uses observed (seed, frame_idx) data points to build a linear model.
    Falls back to naive linear extrapolation when insufficient data.
    """
    # Need at least 2 observations for regression
    if len(seed_frame_map) >= 2:
        seeds = sorted(seed_frame_map.keys())
        frames = [seed_frame_map[s] for s in seeds]
        n = len(seeds)
        sum_s = sum(seeds)
        sum_f = sum(frames)
        sum_sf = sum(s * f for s, f in zip(seeds, frames))
        sum_ss = sum(s * s for s in seeds)
        denom = n * sum_ss - sum_s * sum_s
        if denom != 0:
            slope = (n * sum_sf - sum_s * sum_f) / denom
            intercept = (sum_f - slope * sum_s) / n
            estimate = int(round(slope * seed + intercept))
            return max(0, min(estimate, total_frames - 1))

    # Fallback: naive linear mapping
    return max(0, min(int((seed - 1) * frames_per_qr), total_frames - 1))


def _targeted_recovery(video_path, total_frames, src_fps, workers,
                       seen_seeds, unique_blocks, decoded_count,
                       no_detect_count, lt_decoder, avg_repeat, verbose,
                       seed_frame_map: dict[int, int] | None = None,
                       qr_detector=None):
    """Targeted recovery: read only the video segments where missing seeds
    are expected to appear, scanning every frame in those segments.

    Uses observed (seed, frame_idx) mapping from probe and main scan to
    build a linear model for estimating missing seed positions.
    """
    if seed_frame_map is None:
        seed_frame_map = {}

    # Figure out total encoded block count from the max seed we've seen
    if not seen_seeds:
        return unique_blocks, decoded_count, no_detect_count

    max_seed = max(seen_seeds)
    # frames_per_qr = how many video frames each QR code is shown
    frames_per_qr = max(1, avg_repeat)

    # Identify missing seeds (seeds we never detected)
    all_seeds = set(range(1, max_seed + 1))
    missing_seeds = all_seeds - seen_seeds

    if not missing_seeds:
        return unique_blocks, decoded_count, no_detect_count

    # Calculate frame ranges for missing seeds using observed mapping
    frame_ranges = []
    margin = max(2, int(frames_per_qr * 0.5))  # extra frames for safety

    for seed in sorted(missing_seeds):
        center = _estimate_frame_for_seed(
            seed, seed_frame_map, frames_per_qr, total_frames)
        start = max(0, center - margin)
        end = min(total_frames - 1, center + int(frames_per_qr) + margin)
        frame_ranges.append((start, end))

    # Merge overlapping ranges
    frame_ranges = _merge_ranges(frame_ranges)

    target_frames = sum(e - s + 1 for s, e in frame_ranges)
    print(f"Targeted recovery: {len(missing_seeds)} missing seeds, "
          f"reading {target_frames} frames in {len(frame_ranges)} segments")

    pbar = tqdm(total=target_frames, desc="Recover",
                unit="f", dynamic_ncols=True,
                mininterval=0, miniters=1,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]")

    clahe_worker = partial(_worker_detect_qr_clahe, qr_detector=qr_detector)
    early_done = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        decoded_count, no_detect_count, early_done = _stream_scan(
            executor,
            _read_frame_ranges(video_path, frame_ranges),
            seen_seeds, unique_blocks,
            decoded_count, no_detect_count, lt_decoder, pbar, verbose,
            seed_frame_map, workers,
            worker_fn=clahe_worker)
        if early_done and verbose:
            tqdm.write("  Targeted recovery: all blocks recovered!")

    pbar.close()

    status = " (complete)" if early_done else ""
    print(f"Targeted recovery done{status}: "
          f"{decoded_count} unique blocks, {no_detect_count} missed")

    return unique_blocks, decoded_count, no_detect_count
