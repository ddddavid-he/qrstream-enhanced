"""
LT Fountain Code Decoder: QR video → LT decode → file reconstruction.

Supports V2 protocol with CRC32 validation.
Features adaptive sample rate and targeted frame recovery.
"""

import sys
import io
import zlib
import base64
from math import ceil, log
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import cv2
import numpy as np
from tqdm import tqdm

from .lt_codec import PRNG, BlockGraph, DEFAULT_C, DEFAULT_DELTA
from .protocol import (
    unpack, V2Header, V2_HEADER_SIZE,
)
from .qr_utils import try_decode_qr


class LTDecoder:
    """Consumes LT fountain-coded blocks and reconstructs the original data.

    Accepts V2 blocks with CRC validation; corrupt blocks are silently
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
        seed = header.seed
        compressed = header.compressed

        if compressed:
            self.compressed = True

        if not self.initialized:
            self.filesize = filesize
            self.blocksize = blocksize
            self.K = ceil(filesize / blocksize)
            self.block_graph = BlockGraph(self.K)
            self.prng = PRNG(self.K, delta=self.delta, c=self.c)
            self.initialized = True

        _, _, src_blocks = self.prng.get_src_blocks(seed=seed)

        # Pad data to blocksize if needed
        if len(data) < self.blocksize:
            data = data + b'\x00' * (self.blocksize - len(data))
        elif len(data) > self.blocksize:
            data = data[:self.blocksize]

        self.done = self.block_graph.add_block(src_blocks, data)
        return self.done, self.compressed

    def decode_bytes(self, block_bytes: bytes, skip_crc: bool = False) -> tuple[bool, bool]:
        """Decode a raw V2 block from bytes.

        Validates CRC32 — raises ValueError on corrupt data,
        unless skip_crc=True (for pre-validated blocks).
        """
        header, data = unpack(block_bytes, skip_crc=skip_crc)
        return self.consume_block(header, data)

    def bytes_dump(self) -> bytes:
        """Reconstruct the original file data from recovered blocks.

        Handles both numpy arrays and bytes from the BlockGraph.
        """
        buf = io.BytesIO()
        for ix in range(self.K):
            block = self.block_graph.eliminated.get(ix)
            if block is None:
                raise RuntimeError(
                    f"Missing block {ix}/{self.K} — decoding incomplete")
            # Convert numpy array to bytes if needed
            if isinstance(block, np.ndarray):
                block = block.tobytes()
            if ix < self.K - 1 or self.filesize % self.blocksize == 0:
                buf.write(block)
            else:
                buf.write(block[:self.filesize % self.blocksize])
        raw_data = buf.getvalue()
        if self.compressed:
            return zlib.decompress(raw_data)
        return raw_data


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
    block_bytes = None

    for decode_fn in (_try_base64, lambda d: _try_cobs(d, cobs_decode)):
        candidate = decode_fn(qr_data)
        if candidate is None:
            continue
        # Validate with CRC32
        if len(candidate) >= V2_HEADER_SIZE and candidate[0] == 0x02:
            stored_crc = int.from_bytes(candidate[16:20], 'big')
            computed_crc = (
                zlib.crc32(candidate[:16] + candidate[V2_HEADER_SIZE:])
                & 0xFFFFFFFF
            )
            if computed_crc == stored_crc:
                block_bytes = candidate
                break

    if block_bytes is None:
        return (frame_idx, None, None)

    try:
        if len(block_bytes) >= V2_HEADER_SIZE and block_bytes[0] == 0x02:
            seed = int.from_bytes(block_bytes[10:14], 'big')
        else:
            return (frame_idx, None, None)
        return (frame_idx, block_bytes, seed)
    except Exception:
        return (frame_idx, None, None)


def _try_base64(qr_data: str) -> bytes | None:
    """Try to decode QR payload as base64."""
    try:
        return base64.b64decode(qr_data)
    except Exception:
        return None


def _try_cobs(qr_data: str, cobs_decode_fn) -> bytes | None:
    """Try to decode QR payload as COBS-encoded binary (latin-1 → COBS decode)."""
    try:
        raw = qr_data.encode('latin-1')
        return cobs_decode_fn(raw)
    except Exception:
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


def _probe_sample_rate(video_path: str, workers: int,
                       verbose: bool = False):
    """Probe the first segment of a video to determine optimal sample_rate.

    Measures both repeat count (R) and detection rate (p), then computes
    the optimal sample_rate so that each QR code has enough detection
    chances to be reliably recovered.

    Returns:
        (sample_rate, probe_results, frames_probed, detect_rate, avg_repeat)
    """
    PROBE_FRAMES = 30  # Slightly more than before for better stats

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    probe_count = min(PROBE_FRAMES, total_frames)

    # Read probe frames with sample_rate=1
    probe_frames = list(_read_frames(video_path, 1, total_frames))
    probe_frames = probe_frames[:probe_count]

    if not probe_frames:
        return 1, [], 0, 0.0, 1.0

    # Detect QR codes in probe frames
    probe_results = []
    pbar = tqdm(total=probe_count, desc="Probing sample rate",
                unit="frame", dynamic_ncols=True)
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

    # ── Measure detection rate ─────────────────────────────────────
    detected = sum(1 for _, b, s in probe_results if b is not None)
    detect_rate = detected / probe_count if probe_count > 0 else 0.0

    # ── Measure average repeat count ───────────────────────────────
    seed_runs = []
    current_seed = None
    current_run = 0

    for _, block_bytes, seed in probe_results:
        if seed is not None:
            if seed == current_seed:
                current_run += 1
            else:
                if current_run > 0:
                    seed_runs.append(current_run)
                current_seed = seed
                current_run = 1
        # Failed detection doesn't break runs

    if current_run > 0:
        seed_runs.append(current_run)

    if not seed_runs:
        return 1, probe_results, probe_count, detect_rate, 1.0

    avg_run = sum(seed_runs) / len(seed_runs)

    # ── Compute adaptive sample_rate ───────────────────────────────
    # Goal: for each QR code shown R times, sampling every S frames
    # gives chances = R / S detection opportunities.
    # Probability of detecting at least once = 1 - (1-p)^chances
    # We want this probability >= target (e.g., 0.95).
    #
    # Solve: 1 - (1-p)^(R/S) >= target
    #    →   (1-p)^(R/S) <= 1-target
    #    →   (R/S) * log(1-p) <= log(1-target)
    #    →   R/S >= log(1-target) / log(1-p)
    #    →   S <= R * log(1-p) / log(1-target)

    TARGET_DETECT_PROB = 0.95  # want 95% chance per QR
    p = detect_rate

    if p >= 0.99:
        # Nearly perfect detection — can sample aggressively
        auto_rate = max(1, int(avg_run / 1.5))
    elif p > 0.01:
        min_chances = log(1 - TARGET_DETECT_PROB) / log(1 - p)
        auto_rate = max(1, int(avg_run / min_chances))
    else:
        # Very low detection rate — sample every frame
        auto_rate = 1

    print(f"Probe: {probe_count} frames, detect_rate={detect_rate:.0%}, "
          f"avg_repeat={avg_run:.1f} → sample_rate={auto_rate}")

    return auto_rate, probe_results, probe_count, detect_rate, avg_run


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
        print(f"Error: Cannot open video file: {video_path}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0
    cap.release()

    if workers is None:
        workers = min(multiprocessing.cpu_count(), 8)

    if verbose:
        print(f"Video: {total_frames} frames, {src_fps:.1f} FPS, {duration:.1f}s")
        print(f"Using {workers} worker processes")

    seen_seeds = set()
    unique_blocks = []
    decoded_count = 0
    no_detect_count = 0
    lt_decoder = LTDecoder()

    # ── Auto sample_rate probe ────────────────────────────────────
    probe_results = []
    frames_probed = 0
    detect_rate = 1.0
    avg_repeat = 1.0

    if sample_rate <= 0:
        (auto_rate, probe_results, frames_probed,
         detect_rate, avg_repeat) = _probe_sample_rate(
            video_path, workers, verbose)
        sample_rate = auto_rate
        if verbose:
            print(f"  Using auto sample_rate={sample_rate}")

        # Feed probe results into decoder
        for fidx, block_bytes, seed in probe_results:
            if block_bytes is not None and seed is not None:
                if seed not in seen_seeds:
                    seen_seeds.add(seed)
                    unique_blocks.append(block_bytes)
                    decoded_count += 1
                    try:
                        done, _ = lt_decoder.decode_bytes(block_bytes, skip_crc=True)
                        if done:
                            print(f"Extraction done (during probe): "
                                  f"{frames_probed} frames, "
                                  f"{decoded_count} unique blocks")
                            return unique_blocks
                    except Exception:
                        pass
            else:
                no_detect_count += 1

    if verbose and frames_probed > 0:
        pct = lt_decoder.progress * 100
        print(f"  After probe: {decoded_count} unique blocks, "
              f"progress={pct:.1f}%")

    # ── Main scan (remaining frames) ─────────────────────────────
    BATCH_SIZE = workers * 4
    pbar = tqdm(total=total_frames, desc="Scanning frames",
                unit="frame", dynamic_ncols=True)

    if frames_probed > 0:
        pbar.update(frames_probed)

    last_reported_frame = frames_probed - 1
    early_done = False

    with ProcessPoolExecutor(max_workers=workers) as executor:
        batch = []

        for frame_data in _read_frames(video_path, sample_rate,
                                        total_frames,
                                        start_frame=frames_probed):
            batch.append(frame_data)

            current_frame_idx = frame_data[0]
            skipped = current_frame_idx - last_reported_frame - 1
            if skipped > 0:
                pbar.update(skipped)
            last_reported_frame = current_frame_idx

            if len(batch) >= BATCH_SIZE:
                decoded_count, no_detect_count, early_done = _process_batch(
                    executor, batch, seen_seeds, unique_blocks,
                    decoded_count, no_detect_count, lt_decoder, pbar, verbose)
                batch.clear()
                if early_done:
                    if verbose:
                        tqdm.write(
                            "  Early termination: all source blocks recovered!")
                    break

        if batch and not early_done:
            decoded_count, no_detect_count, early_done = _process_batch(
                executor, batch, seen_seeds, unique_blocks,
                decoded_count, no_detect_count, lt_decoder, pbar, verbose)

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
            lt_decoder, avg_repeat, verbose)

    return unique_blocks


def _targeted_recovery(video_path, total_frames, src_fps, workers,
                       seen_seeds, unique_blocks, decoded_count,
                       no_detect_count, lt_decoder, avg_repeat, verbose):
    """Targeted recovery: read only the video segments where missing seeds
    are expected to appear, scanning every frame in those segments.

    Seeds are sequential (1, 2, ..., N) and the source video plays at
    src_fps. Each seed i appears around frame = (i-1) * frames_per_qr.
    """
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

    # Calculate frame ranges for missing seeds
    # Each seed i occupies frames roughly [(i-1)*fpq, i*fpq)
    frame_ranges = []
    margin = max(2, int(frames_per_qr * 0.5))  # extra frames for safety

    for seed in sorted(missing_seeds):
        center = int((seed - 1) * frames_per_qr)
        start = max(0, center - margin)
        end = min(total_frames - 1, center + int(frames_per_qr) + margin)
        frame_ranges.append((start, end))

    # Merge overlapping ranges
    frame_ranges = _merge_ranges(frame_ranges)

    target_frames = sum(e - s + 1 for s, e in frame_ranges)
    print(f"Targeted recovery: {len(missing_seeds)} missing seeds, "
          f"reading {target_frames} frames in {len(frame_ranges)} segments")

    # Read and process targeted frames
    BATCH_SIZE = workers * 4
    pbar = tqdm(total=target_frames, desc="Targeted recovery",
                unit="frame", dynamic_ncols=True)

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
                   decoded_count, no_detect_count, lt_decoder, pbar, verbose):
    """Submit a batch to the pool and collect results."""
    early_done = False
    futures = {executor.submit(_worker_detect_qr, fd): fd[0] for fd in batch}
    for future in as_completed(futures):
        fidx, block_bytes, seed = future.result()
        pbar.update(1)
        if block_bytes is not None and seed is not None:
            if seed not in seen_seeds:
                seen_seeds.add(seed)
                unique_blocks.append(block_bytes)
                decoded_count += 1

                try:
                    # Workers already validated CRC, skip re-validation
                    done, _ = lt_decoder.decode_bytes(block_bytes, skip_crc=True)
                    if done:
                        early_done = True
                except Exception:
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


def decode_blocks(blocks, verbose=False) -> bytes | None:
    """Feed blocks into LT decoder to reconstruct the file."""
    if not blocks:
        print("Error: No blocks to decode")
        return None

    decoder = LTDecoder()

    for i, block_bytes in enumerate(blocks):
        try:
            done, compressed = decoder.decode_bytes(block_bytes)
            if done:
                if verbose:
                    print(f"  Decoded after {i + 1}/{len(blocks)} blocks "
                          f"(filesize={decoder.filesize}, K={decoder.K}, "
                          f"compressed={compressed})")
                return decoder.bytes_dump()
        except ValueError as e:
            # CRC mismatch on V2 — skip corrupt block
            if verbose:
                print(f"  Block {i} CRC error, skipping: {e}")
        except Exception as e:
            if verbose:
                print(f"  Block {i} error: {e}")

    n_recovered = decoder.num_recovered
    k = decoder.K if decoder.K else '?'
    print(f"\nDecoding incomplete: {n_recovered}/{k} source blocks recovered "
          f"from {len(blocks)} encoded blocks.")
    print("Try recording the QR stream longer to capture more unique frames.")
    return None
