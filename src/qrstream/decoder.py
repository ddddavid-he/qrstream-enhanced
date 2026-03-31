"""
LT Fountain Code Decoder: QR video → LT decode → file reconstruction.

Supports V2/V3 protocols with CRC32 validation.
Features adaptive sample rate and targeted frame recovery.
"""

import io
import struct
import zlib
import base64
from math import ceil, log
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import cv2
import numpy as np
from tqdm import tqdm

from .lt_codec import PRNG, BlockGraph, DEFAULT_C, DEFAULT_DELTA
from .protocol import unpack
from .qr_utils import try_decode_qr


_PROGRESS_BAR_THRESHOLD = 512


class LTDecoder:
    """Consumes LT fountain-coded blocks and reconstructs the original data.

    Accepts V2/V3 blocks with CRC validation; corrupt blocks are silently
    discarded.
    """

    def __init__(self, c: float = DEFAULT_C, delta: float = DEFAULT_DELTA):
        self.c = c
        self.delta = delta
        self.K = 0
        self.filesize = 0
        self.blocksize = 0
        self.done = False
        self.compressed = False
        self.protocol_version = None
        self.block_graph = None
        self.prng = None
        self.initialized = False

    @property
    def progress(self) -> float:
        """Return decoding progress as a fraction [0.0, 1.0]."""
        if not self.initialized or self.K == 0:
            return 0.0
        return min(len(self.block_graph.eliminated) / self.K, 1.0)

    @property
    def num_recovered(self) -> int:
        if self.block_graph is None:
            return 0
        return len(self.block_graph.eliminated)

    def is_done(self) -> bool:
        return self.done

    def consume_block(self, header, data: bytes) -> tuple[bool, bool]:
        """Feed a parsed block (header + data bytes) into the decoder.

        Returns (done, compressed).
        """
        filesize = header.filesize
        blocksize = header.blocksize
        block_count = header.block_count
        seed = header.seed
        compressed = header.compressed

        if blocksize <= 0:
            raise ValueError(f"Invalid blocksize: {blocksize}")

        expected_block_count = ceil(filesize / blocksize) if filesize > 0 else 0
        if block_count != expected_block_count:
            raise ValueError(
                f"block_count mismatch: header={block_count}, expected={expected_block_count}")

        if not self.initialized:
            self.protocol_version = header.version
            self.filesize = filesize
            self.blocksize = blocksize
            self.K = block_count
            self.compressed = compressed
            self.block_graph = BlockGraph(self.K)
            self.prng = PRNG(self.K, delta=self.delta, c=self.c)
            self.initialized = True
        else:
            if header.version != self.protocol_version:
                raise ValueError(
                    f"version mismatch: {header.version} != {self.protocol_version}")
            if filesize != self.filesize:
                raise ValueError(f"filesize mismatch: {filesize} != {self.filesize}")
            if blocksize != self.blocksize:
                raise ValueError(f"blocksize mismatch: {blocksize} != {self.blocksize}")
            if block_count != self.K:
                raise ValueError(f"block_count mismatch: {block_count} != {self.K}")
            if compressed != self.compressed:
                raise ValueError(
                    f"compressed flag mismatch: {compressed} != {self.compressed}")

        _, _, src_blocks = self.prng.get_src_blocks(seed=seed)

        if len(data) < self.blocksize:
            data = data + b'\x00' * (self.blocksize - len(data))
        elif len(data) > self.blocksize:
            data = data[:self.blocksize]

        self.done = self.block_graph.add_block(src_blocks, data)
        return self.done, self.compressed

    def decode_bytes(self, block_bytes: bytes, skip_crc: bool = False) -> tuple[bool, bool]:
        """Decode a raw protocol block from bytes.

        Validates CRC32 — raises ValueError on corrupt data,
        unless skip_crc=True (for pre-validated blocks).
        """
        header, data = unpack(block_bytes, skip_crc=skip_crc)
        return self.consume_block(header, data)

    def _iter_recovered_chunks(self):
        for ix in range(self.K):
            block = self.block_graph.eliminated.get(ix)
            if block is None:
                raise RuntimeError(
                    f"Missing block {ix}/{self.K} — decoding incomplete")
            if isinstance(block, np.ndarray):
                block = block.tobytes()
            if ix < self.K - 1 or self.filesize % self.blocksize == 0:
                yield block
            else:
                yield block[:self.filesize % self.blocksize]

    def bytes_dump(self) -> bytes:
        """Reconstruct the original file data from recovered blocks."""
        buf = io.BytesIO()
        for chunk in self._iter_recovered_chunks():
            buf.write(chunk)
        raw_data = buf.getvalue()
        if self.compressed:
            try:
                return zlib.decompress(raw_data)
            except zlib.error as e:
                raise RuntimeError(
                    f"Decompression failed: {e}. Decoded payload may be corrupted.") from e
        return raw_data

    def bytes_dump_to_file(self, output_path: str, show_progress: bool = False) -> int:
        """Write the reconstructed output directly to a file."""
        written = 0
        pbar = None
        if show_progress:
            pbar = tqdm(total=self.K, desc="Writing output",
                        unit="block", dynamic_ncols=True,
                        mininterval=0.1)

        try:
            with open(output_path, 'wb') as f:
                if self.compressed:
                    decompressor = zlib.decompressobj()
                    try:
                        for chunk in self._iter_recovered_chunks():
                            data = decompressor.decompress(chunk)
                            if data:
                                f.write(data)
                                written += len(data)
                            if pbar is not None:
                                pbar.update(1)
                                pbar.set_postfix(bytes=written)
                        tail = decompressor.flush()
                    except zlib.error as e:
                        raise RuntimeError(
                            f"Decompression failed: {e}. Decoded payload may be corrupted.") from e
                    if tail:
                        f.write(tail)
                        written += len(tail)
                        if pbar is not None:
                            pbar.set_postfix(bytes=written)
                else:
                    for chunk in self._iter_recovered_chunks():
                        f.write(chunk)
                        written += len(chunk)
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix(bytes=written)
        finally:
            if pbar is not None:
                pbar.close()
        return written


# ── Video QR extraction (multiprocess) ───────────────────────────

# Max pixel dimension for QR detection. Frames larger than this are
# downscaled before detection to avoid wasting CPU on 4K+ input.
_MAX_DETECT_DIM = 1080


def _downscale_frame(frame: np.ndarray) -> np.ndarray:
    """Downscale a frame if its larger dimension exceeds _MAX_DETECT_DIM."""
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    if max_dim <= _MAX_DETECT_DIM:
        return frame
    scale = _MAX_DETECT_DIM / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _worker_detect_qr(frame_data):
    """Worker function for multiprocessing QR detection.

    Takes (frame_idx, jpeg_bytes).
    Returns (frame_idx, block_bytes_or_None, seed_or_None).
    """
    from .protocol import cobs_decode

    frame_idx, jpeg_bytes = frame_data
    frame = cv2.imdecode(
        np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return (frame_idx, None, None)

    qr_data = try_decode_qr(frame)

    if qr_data is None:
        return (frame_idx, None, None)

    # Try decoding the QR payload via multiple strategies:
    # 1) base64 (standard mode)
    # 2) latin-1 → COBS decode (binary_qr mode)
    for decode_fn in (_try_base64, lambda d: _try_cobs(d, cobs_decode)):
        candidate = decode_fn(qr_data)
        if candidate is None:
            continue
        try:
            header, _ = unpack(candidate)
            return (frame_idx, candidate, header.seed)
        except (ValueError, struct.error):
            continue

    return (frame_idx, None, None)


def _try_base64(qr_data: str) -> bytes | None:
    """Try to decode QR payload as base64."""
    try:
        return base64.b64decode(qr_data)
    except (ValueError, base64.binascii.Error):
        return None


def _try_cobs(qr_data: str, cobs_decode_fn) -> bytes | None:
    """Try to decode QR payload as COBS-encoded binary (latin-1 → COBS decode)."""
    try:
        raw = qr_data.encode('latin-1')
        return cobs_decode_fn(raw)
    except (ValueError, UnicodeEncodeError):
        return None


def _read_frames(video_path, sample_rate, total_frames, start_frame=0):
    """Generator that reads frames from video.

    Yields (frame_idx, jpeg_bytes) tuples. Frames are downscaled to
    _MAX_DETECT_DIM before JPEG encoding to reduce IPC payload size.
    """
    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % sample_rate == 0:
            # Downscale before encoding to reduce JPEG payload
            frame = _downscale_frame(frame)
            _, jpeg_bytes = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            yield (frame_idx, jpeg_bytes.tobytes())
        frame_idx += 1
    cap.release()


def _read_frame_ranges(video_path, frame_ranges):
    """Generator that reads specific frame ranges from video.

    Args:
        frame_ranges: list of (start_frame, end_frame) tuples (inclusive).

    Yields (frame_idx, jpeg_bytes) tuples for all frames within ranges.
    """
    if not frame_ranges:
        return
    cap = cv2.VideoCapture(video_path)
    for start, end in sorted(frame_ranges):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for fidx in range(start, end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            frame = _downscale_frame(frame)
            _, jpeg_bytes = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            yield (fidx, jpeg_bytes.tobytes())
    cap.release()


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
                       verbose: bool = False):
    """Probe multiple windows of a video to determine optimal sample_rate.

    Measures both repeat count (R) and detection rate (p), then computes
    the optimal sample_rate so that each QR code has enough detection
    chances to be reliably recovered.

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

    # Detect QR codes in probe frames.
    # The probe is short, so force tqdm to refresh every update instead of
    # waiting for the default refresh interval and only rendering at the end.
    probe_results = []
    pbar = tqdm(total=probe_count, desc="Probing sample rate",
                unit="frame", dynamic_ncols=True,
                mininterval=0, miniters=1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker_detect_qr, fd): fd[0]
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


def extract_qr_from_video(video_path: str, sample_rate: int = 0,
                           verbose: bool = False, workers: int | None = None):
    """Extract unique QR code payloads from a video file.

    Uses an LT decoder internally for early termination: stops scanning
    as soon as all source blocks are recovered.

    When initial scan doesn't recover all blocks, performs targeted
    recovery by reading only the video segments corresponding to
    missing seeds.

    Args:
        sample_rate: Process every Nth frame. 0 = auto-detect (default).
        verbose: Print progress details.
        workers: Number of parallel worker processes.

    Returns a list of raw block byte strings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0
    cap.release()

    if workers is None:
        workers = multiprocessing.cpu_count() or 1

    if verbose:
        print(f"Video: {total_frames} frames, {src_fps:.1f} FPS, {duration:.1f}s")
        print(f"Using {workers} worker processes")

    seen_seeds = set()
    unique_blocks = []
    decoded_count = 0
    no_detect_count = 0
    lt_decoder = LTDecoder()
    seed_frame_map: dict[int, int] = {}  # observed seed → first frame index

    # ── Auto sample_rate probe ────────────────────────────────────
    probe_results = []
    probe_count = 0
    leading_frames_probed = 0
    detect_rate = 1.0
    avg_repeat = 1.0

    if sample_rate <= 0:
        (auto_rate, probe_results, probe_count,
         leading_frames_probed, detect_rate, avg_repeat) = _probe_sample_rate(
            video_path, workers, verbose)
        sample_rate = auto_rate
        if verbose:
            print(f"  Using auto sample_rate={sample_rate}")

        # Feed probe results into decoder
        for fidx, block_bytes, seed in probe_results:
            if block_bytes is not None and seed is not None:
                if seed not in seed_frame_map:
                    seed_frame_map[seed] = fidx
                if seed not in seen_seeds:
                    seen_seeds.add(seed)
                    unique_blocks.append(block_bytes)
                    decoded_count += 1
                    try:
                        done, _ = lt_decoder.decode_bytes(block_bytes, skip_crc=True)
                        if done:
                            print(f"Extraction done (during probe): "
                                  f"{probe_count} sampled frames, "
                                  f"{decoded_count} unique blocks")
                            return unique_blocks
                    except (ValueError, struct.error):
                        pass
            else:
                no_detect_count += 1

    if verbose and probe_count > 0:
        pct = lt_decoder.progress * 100
        print(f"  After probe: {decoded_count} unique blocks, "
              f"progress={pct:.1f}%")

    # ── Main scan (remaining frames) ─────────────────────────────
    BATCH_SIZE = workers * 4
    pbar = tqdm(total=total_frames, desc="Scanning frames",
                unit="frame", dynamic_ncols=True)

    if leading_frames_probed > 0:
        pbar.update(leading_frames_probed)

    last_reported_frame = leading_frames_probed - 1
    early_done = False

    with ProcessPoolExecutor(max_workers=workers) as executor:
        batch = []

        for frame_data in _read_frames(video_path, sample_rate,
                                        total_frames,
                                        start_frame=leading_frames_probed):
            batch.append(frame_data)

            current_frame_idx = frame_data[0]
            skipped = current_frame_idx - last_reported_frame - 1
            if skipped > 0:
                pbar.update(skipped)
            last_reported_frame = current_frame_idx

            if len(batch) >= BATCH_SIZE:
                decoded_count, no_detect_count, early_done = _process_batch(
                    executor, batch, seen_seeds, unique_blocks,
                    decoded_count, no_detect_count, lt_decoder, pbar, verbose,
                    seed_frame_map)
                batch.clear()
                if early_done:
                    if verbose:
                        tqdm.write(
                            "  Early termination: all source blocks recovered!")
                    break

        if batch and not early_done:
            decoded_count, no_detect_count, early_done = _process_batch(
                executor, batch, seen_seeds, unique_blocks,
                decoded_count, no_detect_count, lt_decoder, pbar, verbose,
                seed_frame_map)

    remaining = total_frames - (last_reported_frame + 1)
    if remaining > 0:
        pbar.update(remaining)
    pbar.close()

    total_processed = decoded_count + no_detect_count
    status = " (early termination)" if early_done else ""
    print(f"Extraction done{status}: {total_frames} frames "
          f"({total_processed} sampled, sample_rate={sample_rate}), "
          f"{decoded_count} unique blocks, "
          f"{no_detect_count} missed")

    # ── Targeted recovery for missing seeds ───────────────────────
    if (not early_done and lt_decoder.initialized
            and not lt_decoder.done and sample_rate > 1):
        unique_blocks, decoded_count, no_detect_count = _targeted_recovery(
            video_path, total_frames, src_fps, workers,
            seen_seeds, unique_blocks, decoded_count, no_detect_count,
            lt_decoder, avg_repeat, verbose, seed_frame_map)

    return unique_blocks


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
                       seed_frame_map: dict[int, int] | None = None):
    """Targeted recovery: read only the video segments where missing seeds
    are expected to appear, scanning every frame in those segments.

    Uses observed (seed, frame_idx) mapping from probe and main scan to
    build a linear model for estimating missing seed positions. Falls back
    to naive linear estimation when insufficient observations are available.
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

    # Read and process targeted frames.
    # Force frequent refreshes and explicitly include percent so the progress
    # bar remains informative even for short targeted scans.
    BATCH_SIZE = workers * 4
    pbar = tqdm(total=target_frames, desc="Targeted recovery",
                unit="frame", dynamic_ncols=True,
                mininterval=0, miniters=1,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]")

    early_done = False
    with ProcessPoolExecutor(max_workers=workers) as executor:
        batch = []
        for frame_data in _read_frame_ranges(video_path, frame_ranges):
            batch.append(frame_data)
            if len(batch) >= BATCH_SIZE:
                decoded_count, no_detect_count, early_done = _process_batch(
                    executor, batch, seen_seeds, unique_blocks,
                    decoded_count, no_detect_count, lt_decoder, pbar, verbose)
                batch.clear()
                if early_done:
                    if verbose:
                        tqdm.write(
                            "  Targeted recovery: all blocks recovered!")
                    break

        if batch and not early_done:
            decoded_count, no_detect_count, early_done = _process_batch(
                executor, batch, seen_seeds, unique_blocks,
                decoded_count, no_detect_count, lt_decoder, pbar, verbose)

    pbar.close()

    status = " (complete)" if early_done else ""
    print(f"Targeted recovery done{status}: "
          f"{decoded_count} unique blocks, {no_detect_count} missed")

    return unique_blocks, decoded_count, no_detect_count


def _merge_ranges(ranges):
    """Merge overlapping or adjacent (start, end) ranges."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _process_batch(executor, batch, seen_seeds, unique_blocks,
                   decoded_count, no_detect_count, lt_decoder, pbar, verbose,
                   seed_frame_map: dict[int, int] | None = None):
    """Submit a batch to the pool and collect results."""
    early_done = False
    futures = {executor.submit(_worker_detect_qr, fd): fd[0] for fd in batch}
    for future in as_completed(futures):
        fidx, block_bytes, seed = future.result()
        pbar.update(1)
        if block_bytes is not None and seed is not None:
            if seed_frame_map is not None and seed not in seed_frame_map:
                seed_frame_map[seed] = fidx
            if seed not in seen_seeds:
                seen_seeds.add(seed)
                unique_blocks.append(block_bytes)
                decoded_count += 1

                try:
                    # Workers already validated CRC, skip re-validation
                    done, _ = lt_decoder.decode_bytes(block_bytes, skip_crc=True)
                    if done:
                        early_done = True
                except (ValueError, struct.error):
                    pass

                if verbose:
                    pct = lt_decoder.progress * 100
                    tqdm.write(
                        f"  Frame {fidx}: seed={seed}, "
                        f"unique={decoded_count}, "
                        f"progress={pct:.1f}%")
        else:
            no_detect_count += 1
        pbar.set_postfix(unique=decoded_count, missed=no_detect_count)
    return decoded_count, no_detect_count, early_done


def _decode_into_decoder(blocks, verbose=False) -> LTDecoder | None:
    if not blocks:
        print("Error: No blocks to decode")
        return None

    decoder = LTDecoder()
    show_progress = verbose or len(blocks) >= _PROGRESS_BAR_THRESHOLD
    pbar = None
    if show_progress:
        pbar = tqdm(total=len(blocks), desc="Decoding LT blocks",
                    unit="block", dynamic_ncols=True,
                    mininterval=0.1)

    try:
        for i, block_bytes in enumerate(blocks):
            try:
                done, compressed = decoder.decode_bytes(block_bytes)
                if pbar is not None:
                    pbar.update(1)
                    if decoder.initialized:
                        pbar.set_postfix(recovered=decoder.num_recovered, total=decoder.K)
                if done:
                    if verbose:
                        print(f"  Decoded after {i + 1}/{len(blocks)} blocks "
                              f"(filesize={decoder.filesize}, K={decoder.K}, "
                              f"compressed={compressed}, v={decoder.protocol_version})")
                    return decoder
            except ValueError as e:
                if pbar is not None:
                    pbar.update(1)
                    if decoder.initialized:
                        pbar.set_postfix(recovered=decoder.num_recovered, total=decoder.K)
                if verbose:
                    print(f"  Block {i} error, skipping: {e}")
            except Exception as e:
                if pbar is not None:
                    pbar.update(1)
                    if decoder.initialized:
                        pbar.set_postfix(recovered=decoder.num_recovered, total=decoder.K)
                if verbose:
                    print(f"  Block {i} error: {e}")
    finally:
        if pbar is not None:
            pbar.close()

    n_recovered = decoder.num_recovered
    k = decoder.K if decoder.K else '?'
    print(f"\nDecoding incomplete: {n_recovered}/{k} source blocks recovered "
          f"from {len(blocks)} encoded blocks.")
    print("Try recording the QR stream longer to capture more unique frames.")
    return None


def decode_blocks(blocks, verbose=False) -> bytes | None:
    """Feed blocks into LT decoder to reconstruct the file."""
    decoder = _decode_into_decoder(blocks, verbose=verbose)
    if decoder is None:
        return None
    try:
        return decoder.bytes_dump()
    except RuntimeError as e:
        print(f"Error: {e}")
        return None


def decode_blocks_to_file(blocks, output_path: str, verbose=False) -> int | None:
    """Decode blocks and write the result directly to a file."""
    decoder = _decode_into_decoder(blocks, verbose=verbose)
    if decoder is None:
        return None
    try:
        show_progress = verbose or (decoder.K >= _PROGRESS_BAR_THRESHOLD)
        return decoder.bytes_dump_to_file(output_path, show_progress=show_progress)
    except RuntimeError as e:
        print(f"Error: {e}")
        return None
