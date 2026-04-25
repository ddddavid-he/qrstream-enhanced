"""
LT Fountain Code Decoder: QR video → LT decode → file reconstruction.

Supports V2/V3 protocols with CRC32 validation.
Features adaptive sample rate and targeted frame recovery.
"""

import io
import os
import struct
import zlib
import base64
from math import ceil, log
from queue import Queue
from threading import Thread
from concurrent.futures import (
    Executor,
    ThreadPoolExecutor,
    as_completed,
    FIRST_COMPLETED,
    wait as _futures_wait,
)


import cv2
import numpy as np
from tqdm import tqdm

from .lt_codec import PRNG, BlockGraph, DEFAULT_C, DEFAULT_DELTA
from .protocol import unpack
from .qr_utils import try_decode_qr, DETECTOR_CAN_CRASH
from . import qr_sandbox


_PROGRESS_BAR_THRESHOLD = 512

# ── crash-isolation dispatch hook ────────────────────────────────
# Worker functions call ``_dispatch_detect`` instead of
# ``try_decode_qr`` directly. :func:`extract_qr_from_video` swaps
# this to :meth:`qr_sandbox.SandboxedDetector.detect` when
# ``detect_isolation != 'off'`` and restores it on exit, so the
# sandbox is transparent to ``_worker_detect_qr`` /
# ``_worker_detect_qr_clahe``.


def _in_process_detect(_frame_idx: int, frame: "np.ndarray") -> str | None:
    return try_decode_qr(frame)


_dispatch_detect = _in_process_detect


def _validate_isolation_mode(mode: str) -> None:
    if mode not in ("on", "off"):
        raise ValueError(
            f"detect_isolation must be 'on' or 'off', got {mode!r}"
        )

# Maximum frames the reader thread may prefetch ahead of the worker pool
# for the main scan / targeted recovery.  Kept small so we don't balloon
# memory usage when decode falls behind frame-read (e.g. on files where
# no QR codes are detectable and the pool is idle anyway).
_READER_QUEUE_CAPACITY = 64


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
        self.prng_version = None  # set from the first block's header
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
            self.prng_version = header.prng_version
            self.filesize = filesize
            self.blocksize = blocksize
            self.K = block_count
            self.compressed = compressed
            self.block_graph = BlockGraph(self.K)
            self.prng = PRNG(self.K, delta=self.delta, c=self.c,
                             prng_version=self.prng_version)
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
            if header.prng_version != self.prng_version:
                # Mixing prng_version=0 and =1 blocks in the same
                # session is unsolvable: the two PRNG schedules
                # produce entirely different (degree, src_blocks)
                # tuples for the same seed. A well-formed video
                # always has a consistent flag bit across frames.
                raise ValueError(
                    f"prng_version mismatch: {header.prng_version} "
                    f"!= {self.prng_version}")

        _, _, src_blocks = self.prng.get_src_blocks(seed=seed)

        if len(data) < self.blocksize:
            data = data + b'\x00' * (self.blocksize - len(data))
        elif len(data) > self.blocksize:
            data = data[:self.blocksize]

        self.done = self.block_graph.add_block(src_blocks, data)
        return self.done, self.compressed

    def try_gaussian_rescue(self) -> bool:
        """Opt-in GF(2) Gauss-Jordan pass over the current check-node
        graph.

        Call this *after* all available blocks have been fed and
        :meth:`is_done` still returns False.  When the surviving
        check equations together span the missing source blocks,
        this recovers the whole file without needing any more
        encoded frames.  Safe no-op when peeling already converged.

        Returns True iff every source block is now recovered.
        """
        if not self.initialized or self.block_graph is None:
            return False
        if self.done:
            return True
        recovered = self.block_graph.try_gaussian_rescue()
        if recovered:
            self.done = True
        return recovered

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
            pbar = tqdm(total=self.K, desc="Write",
                        unit="blk", dynamic_ncols=True,
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


# ── Video QR extraction (thread pool) ────────────────────────────

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
    """Worker function for thread-pool QR detection.

    Takes (frame_idx, frame_ndarray).
    Returns (frame_idx, block_bytes_or_None, seed_or_None).

    The frame is a ``numpy.ndarray`` (BGR uint8, already downscaled
    to ``_MAX_DETECT_DIM``) handed to the worker by reference: under
    ``ThreadPoolExecutor`` workers share the main process address
    space, so the ndarray travels as a zero-copy reference. The
    per-thread ``WeChatQRCode`` detector is cached in
    :mod:`qrstream.qr_utils`' ``threading.local()``.
    """
    from .protocol import base45_decode, cobs_decode

    frame_idx, frame = frame_data
    if frame is None:
        return (frame_idx, None, None)

    qr_data = _dispatch_detect(frame_idx, frame)

    if qr_data is None:
        return (frame_idx, None, None)

    # Try decoding the QR payload via multiple strategies.  The flag
    # bit 0x02 in the packed header tells us whether the payload is
    # high-density encoded; however we don't have the header yet here
    # (it lives inside the payload), so we try all strategies in
    # order.  Strategies are cheap: each failed attempt is a single
    # ASCII lookup + a constant-time rejection.
    #   1) base45  (current default for high-density mode)
    #   2) base64  (standard mode)
    #   3) COBS/latin-1  (legacy pre-0.6 high-density mode)
    for decode_fn in (
        lambda d: _try_base45(d, base45_decode),
        _try_base64,
        lambda d: _try_cobs(d, cobs_decode),
    ):
        candidate = decode_fn(qr_data)
        if candidate is None:
            continue
        try:
            header, _ = unpack(candidate)
            return (frame_idx, candidate, header.seed)
        except (ValueError, struct.error):
            continue

    return (frame_idx, None, None)


def _worker_detect_qr_clahe(frame_data):
    """Recovery worker: run WeChat on a CLAHE-boosted copy of the frame.

    Used by ``_targeted_recovery`` after the main scan failed to
    deliver enough unique seeds for LT peeling to converge.  CLAHE
    (Contrast Limited Adaptive Histogram Equalisation) is a purely
    scalar, per-tile operation — it does not depend on OpenCV's
    INTER_AREA SIMD dispatch, which is the root cause of why
    ``ubuntu-latest`` amd64 and ``ubuntu-24.04-arm`` disagree about
    which phone-captured frames are "detectable".  By boosting local
    contrast on the QR modules we lift edge frames that got pushed
    just below the WeChatQRCode classifier threshold back above it,
    which is enough to pull the observed seed subset out of LT's
    (rare, ~3%) pathological region.

    Takes ``(frame_idx, frame_ndarray)``. Returns
    ``(frame_idx, block_bytes_or_None, seed_or_None)``.
    """
    frame_idx, frame = frame_data
    if frame is None:
        return (frame_idx, None, None)

    try:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ycrcb[:, :, 0] = clahe.apply(y)
        boosted = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    except cv2.error:
        return (frame_idx, None, None)

    if not boosted.flags['C_CONTIGUOUS']:
        boosted = np.ascontiguousarray(boosted)

    qr_data = _dispatch_detect(frame_idx, boosted)
    if qr_data is None:
        return (frame_idx, None, None)

    # Mirror the multi-strategy decode used by ``_worker_detect_qr``.
    from .protocol import base45_decode, cobs_decode
    for decode_fn in (
        lambda d: _try_base45(d, base45_decode),
        _try_base64,
        lambda d: _try_cobs(d, cobs_decode),
    ):
        candidate = decode_fn(qr_data)
        if candidate is None:
            continue
        try:
            header, _ = unpack(candidate)
            return (frame_idx, candidate, header.seed)
        except (ValueError, struct.error):
            continue

    return (frame_idx, None, None)


def _try_base45(qr_data: str, base45_decode_fn) -> bytes | None:
    """Try to decode QR payload as a base45 (alphanumeric-mode) string."""
    try:
        return base45_decode_fn(qr_data)
    except (ValueError, KeyError):
        return None


def _try_base64(qr_data: str) -> bytes | None:
    """Try to decode QR payload as base64."""
    try:
        return base64.b64decode(qr_data)
    except (ValueError, base64.binascii.Error):
        return None


def _try_cobs(qr_data: str, cobs_decode_fn) -> bytes | None:
    """Try to decode QR payload as COBS-encoded binary (latin-1 → COBS decode).

    Retained for backward compatibility with videos produced by
    pre-0.6 qrstream releases.
    """
    try:
        raw = qr_data.encode('latin-1')
        return cobs_decode_fn(raw)
    except (ValueError, UnicodeEncodeError):
        return None


def _read_frames(video_path, sample_rate, total_frames, start_frame=0):
    """Generator that reads frames from video.

    Yields ``(frame_idx, frame_ndarray)`` tuples. Frames are
    downscaled to ``_MAX_DETECT_DIM`` and passed as BGR uint8
    ndarrays directly.  Worker threads share the main process
    address space, so the ndarray is handed over as a zero-copy
    reference.

    Thread-safety note: ``cv2.VideoCapture.read()`` reuses an
    internal frame buffer — each call returns an ndarray that
    views the *same* memory overwritten on the next iteration.
    Under a ``ThreadPoolExecutor`` a worker can see the live
    buffer scribbled over mid-detect by the producer's next
    ``read()``, which corrupts WeChat's output.  We therefore
    force a contiguous *copy* before yielding so each worker
    owns its frame outright.  ``np.ascontiguousarray`` alone is
    not enough: if the array is already contiguous it returns
    the same object without copying, so we chain ``.copy()``.
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
            frame = _downscale_frame(frame)
            frame = np.ascontiguousarray(frame).copy()
            yield (frame_idx, frame)
        frame_idx += 1
    cap.release()


def _read_frame_ranges(video_path, frame_ranges):
    """Generator that reads specific frame ranges from video.

    Args:
        frame_ranges: list of (start_frame, end_frame) tuples (inclusive).

    Yields ``(frame_idx, frame_ndarray)`` tuples for all frames
    within ranges. See ``_read_frames`` for the rationale behind
    the ndarray (rather than encoded-bytes) payload and the
    mandatory ``.copy()`` (VideoCapture reuses its internal
    buffer; threads would otherwise race with the next ``read()``).
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
            frame = np.ascontiguousarray(frame).copy()
            yield (fidx, frame)
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
    pbar = tqdm(total=probe_count, desc="Probe",
                unit="f", dynamic_ncols=True,
                mininterval=0, miniters=1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
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
                           verbose: bool = False, workers: int | None = None,
                           *, detect_isolation: str = "on"):
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
        detect_isolation: ``'on'`` (default) runs QR detection in a pool
            of subprocess helpers so a native crash in
            ``cv2.wechat_qrcode_WeChatQRCode`` (see
            ``opencv_contrib#3570``) degrades to a single dropped frame
            instead of killing the decode process. ``'off'`` runs
            detection in-process (slightly faster but unsafe on
            camera-captured inputs).

    Returns a list of raw block byte strings.
    """
    global _dispatch_detect
    _validate_isolation_mode(detect_isolation)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0
    cap.release()

    if workers is None:
        workers = os.cpu_count() or 1

    if verbose:
        print(f"Video: {total_frames} frames, {src_fps:.1f} FPS, {duration:.1f}s")
        print(f"Using {workers} worker processes")

    sandbox = None
    original_dispatch = _dispatch_detect
    if detect_isolation == "on":
        try:
            sandbox = qr_sandbox.SandboxedDetector(pool_size=3)
            _dispatch_detect = sandbox.detect
        except Exception as exc:
            print(
                f"[sandbox] failed to initialise ({exc}); "
                f"falling back to in-process detection."
            )
            sandbox = None
    # else: 'off' → stay with _in_process_detect

    try:
        seen_seeds = set()
        unique_blocks = []
        decoded_count = 0
        no_detect_count = 0
        lt_decoder = LTDecoder()
        seed_frame_map: dict[int, int] = {}  # observed seed → first frame index

        # ── Auto sample_rate probe ────────────────────────────────
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

        # ── Main scan (remaining frames) ─────────────────────────
        pbar = tqdm(total=total_frames, desc="Scan",
                    unit="f", dynamic_ncols=True)

        if leading_frames_probed > 0:
            pbar.update(leading_frames_probed)

        early_done = False

        # Wrap _read_frames so pbar updates reflect the current video
        # position even when frames are skipped by sample_rate.
        def _tracking_frame_iter():
            last_reported = leading_frames_probed - 1
            for frame_data in _read_frames(
                    video_path, sample_rate, total_frames,
                    start_frame=leading_frames_probed):
                skipped = frame_data[0] - last_reported - 1
                if skipped > 0:
                    pbar.update(skipped)
                last_reported = frame_data[0]
                yield frame_data
            # After iteration, advance pbar for any frames past the last
            # sampled one (they were read but not yielded).
            remaining = total_frames - (last_reported + 1)
            if remaining > 0:
                pbar.update(remaining)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            decoded_count, no_detect_count, early_done = _stream_scan(
                executor, _tracking_frame_iter(),
                seen_seeds, unique_blocks,
                decoded_count, no_detect_count, lt_decoder, pbar, verbose,
                seed_frame_map, workers)
            if early_done and verbose:
                tqdm.write(
                    "  Early termination: all source blocks recovered!")

        pbar.close()

        total_processed = decoded_count + no_detect_count
        status = " (early termination)" if early_done else ""
        print(f"Extraction done{status}: {total_frames} frames "
              f"({total_processed} sampled, sample_rate={sample_rate}), "
              f"{decoded_count} unique blocks, "
              f"{no_detect_count} missed")

        # ── Targeted recovery for missing seeds ───────────────────
        # Triggered whenever the main scan finished without LT converging,
        # regardless of ``sample_rate``.  The previous ``sample_rate > 1``
        # guard skipped recovery on videos where the probe decided to read
        # every frame (sample_rate=1) — but such a video can still land on
        # a pathological ~3% LT seed subset (see v070 amd64 regression)
        # and recovery has a cheap CLAHE-boosted rescan to offer even when
        # the main scan already visited every frame.
        if (not early_done and lt_decoder.initialized
                and not lt_decoder.done):
            unique_blocks, decoded_count, no_detect_count = _targeted_recovery(
                video_path, total_frames, src_fps, workers,
                seen_seeds, unique_blocks, decoded_count, no_detect_count,
                lt_decoder, avg_repeat, verbose, seed_frame_map)

        return unique_blocks
    finally:
        _dispatch_detect = original_dispatch
        if sandbox is not None:
            crashes = sandbox.crash_count
            sandbox.close()
            if crashes > 0:
                # Unconditional print (not gated on --verbose).
                print(
                    f"[sandbox] detector crashed {crashes} time(s) "
                    f"during decode; affected frames treated as "
                    f"no-detect. Decoding proceeded normally."
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
    pbar = tqdm(total=target_frames, desc="Recover",
                unit="f", dynamic_ncols=True,
                mininterval=0, miniters=1,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]")

    early_done = False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        decoded_count, no_detect_count, early_done = _stream_scan(
            executor,
            _read_frame_ranges(video_path, frame_ranges),
            seen_seeds, unique_blocks,
            decoded_count, no_detect_count, lt_decoder, pbar, verbose,
            seed_frame_map, workers,
            worker_fn=_worker_detect_qr_clahe)
        if early_done and verbose:
            tqdm.write("  Targeted recovery: all blocks recovered!")

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


def _prefetch_iter(source_iter, capacity: int = _READER_QUEUE_CAPACITY):
    """Run ``source_iter`` in a background thread, yielding items in order.

    This lets frame read + downscale overlap with worker-pool
    detection on the main thread, instead of the pre-v0.7 "read a
    batch -> submit -> wait -> next batch" cycle.  Order is
    preserved because there is a single producer and a single
    consumer on a FIFO Queue.

    If the consumer bails out early (via generator .close() /
    GeneratorExit), the producer is notified via ``stop_event`` and
    exits on the next queue put, so it does not keep reading the
    entire video file for nothing.
    """
    from threading import Event

    _SENTINEL = object()
    q: Queue = Queue(maxsize=capacity)
    stop_event = Event()

    def _producer():
        try:
            for item in source_iter:
                if stop_event.is_set():
                    return
                q.put(item)
        finally:
            q.put(_SENTINEL)

    t = Thread(target=_producer, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is _SENTINEL:
                return
            yield item
    finally:
        # Ask the producer to stop reading on the next iteration; then
        # drain the queue so a put-blocked producer can unblock and
        # see the flag.
        stop_event.set()
        while t.is_alive():
            try:
                item = q.get(timeout=0.1)
                if item is _SENTINEL:
                    break
            except Exception:
                break


def _stream_scan(executor: Executor, frame_iter, seen_seeds, unique_blocks,
                 decoded_count, no_detect_count, lt_decoder, pbar, verbose,
                 seed_frame_map, workers, worker_fn=None):
    """Pipelined scan: keep ``workers*2`` detect tasks in flight at all times.

    Reads frames via ``_prefetch_iter`` (background thread) and feeds
    them to ``executor`` using a sliding window of pending futures.
    Each completed future updates ``pbar`` by 1, so progress visibly
    advances frame-by-frame instead of in batch-sized jumps.

    ``worker_fn`` defaults to :func:`_worker_detect_qr` (plain WeChat
    detection on the already-downscaled frame).  Targeted recovery
    passes :func:`_worker_detect_qr_clahe` to rescue frames the main
    scan missed by the ε-margin introduced by cross-architecture
    ``cv2.resize(INTER_AREA)`` SIMD drift.
    """
    if worker_fn is None:
        worker_fn = _worker_detect_qr

    early_done = False
    IN_FLIGHT = max(workers * 2, 4)

    prefetched = _prefetch_iter(frame_iter)
    pending: set = set()

    def _submit_next() -> bool:
        """Pull one frame and submit it. Return False when exhausted."""
        try:
            fd = next(prefetched)
        except StopIteration:
            return False
        pending.add(executor.submit(worker_fn, fd))
        return True

    # Prime the pool
    for _ in range(IN_FLIGHT):
        if not _submit_next():
            break

    while pending and not early_done:
        done_set, pending = _futures_wait(pending, return_when=FIRST_COMPLETED)
        for fut in done_set:
            fidx, block_bytes, seed = fut.result()
            pbar.update(1)
            if block_bytes is not None and seed is not None:
                if seed_frame_map is not None and seed not in seed_frame_map:
                    seed_frame_map[seed] = fidx
                if seed not in seen_seeds:
                    seen_seeds.add(seed)
                    unique_blocks.append(block_bytes)
                    decoded_count += 1
                    try:
                        done, _ = lt_decoder.decode_bytes(
                            block_bytes, skip_crc=True)
                        if done:
                            early_done = True
                    except (ValueError, struct.error):
                        pass
                    if verbose:
                        pct = lt_decoder.progress * 100
                        tqdm.write(
                            f"  Frame {fidx}: seed={seed}, "
                            f"uniq={decoded_count}, "
                            f"progress={pct:.1f}%")
            else:
                no_detect_count += 1
            total_seen = decoded_count + no_detect_count
            hit_pct = (decoded_count * 100 // total_seen) if total_seen else 0
            pbar.set_postfix_str(f"hit={hit_pct}%, uniq={decoded_count}")

            # Keep the pool topped up — one in, one out.
            if not early_done:
                _submit_next()

    # On early termination, cancel anything still queued so we release
    # the executor promptly.
    for fut in pending:
        fut.cancel()

    return decoded_count, no_detect_count, early_done


def _decode_into_decoder(blocks, verbose=False) -> LTDecoder | None:
    if not blocks:
        print("Error: No blocks to decode")
        return None

    decoder = LTDecoder()
    show_progress = verbose or len(blocks) >= _PROGRESS_BAR_THRESHOLD
    pbar = None
    if show_progress:
        pbar = tqdm(total=len(blocks), desc="LT decode",
                    unit="blk", dynamic_ncols=True,
                    mininterval=0.1)

    try:
        for i, block_bytes in enumerate(blocks):
            try:
                done, compressed = decoder.decode_bytes(block_bytes)
                if pbar is not None:
                    pbar.update(1)
                    if decoder.initialized:
                        pbar.set_postfix_str(
                            f"got={decoder.num_recovered}/{decoder.K}")
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
                        pbar.set_postfix_str(
                            f"got={decoder.num_recovered}/{decoder.K}")
                if verbose:
                    print(f"  Block {i} error, skipping: {e}")
            except Exception as e:
                if pbar is not None:
                    pbar.update(1)
                    if decoder.initialized:
                        pbar.set_postfix_str(
                            f"got={decoder.num_recovered}/{decoder.K}")
                if verbose:
                    print(f"  Block {i} error: {e}")
    finally:
        if pbar is not None:
            pbar.close()

    # Peeling (belief-propagation) exhausted all blocks without
    # converging. Attempt a GF(2) Gauss-Jordan rescue pass over the
    # accumulated check-node graph: if the surviving equations
    # collectively span the missing source blocks, we still get a
    # perfect reconstruction.  This path is only entered on peeling
    # failure, so it costs nothing in the healthy case.
    #
    # TODO(v0.10.0): the main reason peeling fails on a post-0.8
    # stream is legacy prng_version=0 encoding. Once v0 support is
    # dropped (see ``protocol.py``), revisit whether the rescue is
    # still worth carrying — native v1 streams converge above the
    # CLI's ``_MIN_OVERHEAD`` floor, so GE would only help
    # overhead-below-floor edge cases.
    if decoder.initialized and not decoder.done:
        rescued = decoder.try_gaussian_rescue()
        if rescued:
            if verbose:
                print(f"  GE rescue recovered all "
                      f"{decoder.num_recovered}/{decoder.K} blocks "
                      f"after peeling stalled.")
            else:
                print(f"  GE rescue recovered "
                      f"{decoder.num_recovered}/{decoder.K} source blocks.")
            return decoder
        elif verbose:
            print(f"  GE rescue attempted, still "
                  f"{decoder.num_recovered}/{decoder.K} recovered.")

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
