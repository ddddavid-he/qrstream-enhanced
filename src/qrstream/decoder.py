"""
LT Fountain Code Decoder: QR video → LT decode → file reconstruction.

Supports both V1 (legacy) and V2 (with CRC32) protocol formats.
Uses shared memory for zero-copy frame transfer to worker processes.
"""

import sys
import io
import zlib
import base64
from struct import unpack as struct_unpack
from math import ceil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import cv2
import numpy as np
from tqdm import tqdm

from .lt_codec import PRNG, BlockGraph, DEFAULT_C, DEFAULT_DELTA
from .protocol import (
    unpack, V1Header, V2Header, V1_HEADER_SIZE, V2_HEADER_SIZE,
)
from .qr_utils import try_decode_qr


class LTDecoder:
    """Consumes LT fountain-coded blocks and reconstructs the original data.

    Accepts both V1 and V2 blocks. V2 blocks are CRC-validated; corrupt
    blocks are silently discarded.
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
        if isinstance(header, V2Header):
            filesize = header.filesize
            blocksize = header.blocksize
            seed = header.seed
            compressed = header.compressed
        elif isinstance(header, V1Header):
            filesize = header.filesize
            blocksize = header.blocksize
            seed = header.seed
            compressed = header.compressed
        else:
            raise TypeError(f"Unknown header type: {type(header)}")

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
        """Decode a raw block from bytes (auto-detects V1/V2).

        For V2, validates CRC32 — raises ValueError on corrupt data,
        unless skip_crc=True (for pre-validated blocks).
        """
        header, data = unpack(block_bytes, skip_crc=skip_crc)
        return self.consume_block(header, data)

    def decode_bytes_v1_compat(self, block_bytes: bytes) -> tuple[bool, bool]:
        """Decode a V1 block using the old int-based path for backward compat.

        This method handles V1 blocks that were encoded with int representation.
        """
        header_tuple = struct_unpack('!BIII', block_bytes[:V1_HEADER_SIZE])
        magic_byte, filesize, blocksize, seed = header_tuple

        compressed = bool(magic_byte & 0x01)
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

        # V1 data: raw bytes after 13-byte header, pad to blocksize
        data = block_bytes[V1_HEADER_SIZE:]
        if len(data) < self.blocksize:
            data = data + b'\x00' * (self.blocksize - len(data))
        elif len(data) > self.blocksize:
            data = data[:self.blocksize]

        self.done = self.block_graph.add_block(src_blocks, data)
        return self.done, self.compressed

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

# Shared memory was removed — downscaling in reader + JPEG is simpler and leak-free.


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

    # Reuse QRCodeDetector to avoid repeated object creation
    if not hasattr(_worker_detect_qr, '_qr_detector'):
        _worker_detect_qr._qr_detector = cv2.QRCodeDetector()
    qr_data = try_decode_qr(frame, _worker_detect_qr._qr_detector)

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
        elif len(candidate) >= V1_HEADER_SIZE:
            # V1 has no CRC, accept it
            block_bytes = candidate
            break

    if block_bytes is None:
        return (frame_idx, None, None)

    try:
        if len(block_bytes) >= V2_HEADER_SIZE and block_bytes[0] == 0x02:
            seed = int.from_bytes(block_bytes[10:14], 'big')
        elif len(block_bytes) >= V1_HEADER_SIZE:
            seed = int.from_bytes(block_bytes[9:13], 'big')
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


def _probe_sample_rate(video_path: str, workers: int,
                       verbose: bool = False):
    """Probe the first segment of a video to determine optimal sample_rate.

    Scans frames with sample_rate=1, counts how many consecutive frames
    share the same seed, and returns (auto_sample_rate, probe_results).

    probe_results is a list of (frame_idx, block_bytes, seed) from the probe
    phase that should be reused (not re-scanned).

    Returns:
        (sample_rate, probe_results, frames_probed)
    """
    PROBE_FRAMES = 20  # Probe first ~0.3s at 60fps (sufficient for sample rate)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    probe_count = min(PROBE_FRAMES, total_frames)

    # Read probe frames with sample_rate=1
    probe_frames = list(_read_frames(video_path, 1, total_frames))
    probe_frames = probe_frames[:probe_count]

    if not probe_frames:
        return 1, [], 0

    # Detect QR codes in probe frames with progress bar
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

    # Count consecutive frames with the same seed
    seed_runs = []  # lengths of consecutive runs of the same seed
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
        else:
            # Failed detection — don't break the run, could be a blurry frame
            pass

    if current_run > 0:
        seed_runs.append(current_run)

    if not seed_runs:
        # No QR codes detected at all in probe
        return 1, probe_results, probe_count

    avg_run = sum(seed_runs) / len(seed_runs)

    # Use half the average run as sample_rate (conservative).
    # This ensures we sample each QR code at least twice on average,
    # reducing the chance of missing critical low-degree LT blocks.
    auto_rate = max(1, int(avg_run / 2))

    print(f"Probe complete: {probe_count} frames, "
          f"avg repeat={avg_run:.1f}, auto sample_rate={auto_rate}")

    return auto_rate, probe_results, probe_count


def extract_qr_from_video(video_path: str, sample_rate: int = 0,
                           verbose: bool = False, workers: int | None = None):
    """Extract unique QR code payloads from a video file.

    Uses an LT decoder internally for early termination: stops scanning
    as soon as all source blocks are recovered.

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

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    if workers is None:
        workers = min(multiprocessing.cpu_count(), 8)

    if verbose:
        print(f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
        print(f"Using {workers} worker processes")

    seen_seeds = set()
    unique_blocks = []
    decoded_count = 0
    no_detect_count = 0
    lt_decoder = LTDecoder()

    # ── Auto sample_rate probe ────────────────────────────────────
    probe_results = []
    frames_probed = 0

    if sample_rate <= 0:
        auto_rate, probe_results, frames_probed = _probe_sample_rate(
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
                        if block_bytes[0] == 0x02:
                            done, _ = lt_decoder.decode_bytes(block_bytes, skip_crc=True)
                        else:
                            done, _ = lt_decoder.decode_bytes_v1_compat(
                                block_bytes)
                        if done:
                            # All blocks recovered during probe!
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

    # Account for already-probed frames
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

    return unique_blocks


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
                    # Try V2 first, fall back to V1 compat
                    # Workers already validated CRC, skip re-validation
                    if block_bytes[0] == 0x02:
                        done, _ = lt_decoder.decode_bytes(block_bytes, skip_crc=True)
                    else:
                        done, _ = lt_decoder.decode_bytes_v1_compat(
                            block_bytes)
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
            # Auto-detect V1/V2
            if block_bytes[0] == 0x02:
                done, compressed = decoder.decode_bytes(block_bytes)
            else:
                done, compressed = decoder.decode_bytes_v1_compat(block_bytes)

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
