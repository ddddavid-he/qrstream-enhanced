"""
Video I/O and QR detection workers for the decode pipeline.

Contains frame reading, downscaling, QR payload parsing, CLAHE
enhancement, and the pipelined stream-scan executor.

Split from ``decoder.py`` for readability; all public symbols are
re-exported by ``decoder.py`` so external imports are unchanged.
"""

from __future__ import annotations

import base64
import struct
from functools import partial
from queue import Queue
from threading import Thread
from concurrent.futures import (
    Executor,
    FIRST_COMPLETED,
    wait as _futures_wait,
)

import cv2
import numpy as np
from tqdm import tqdm

from .protocol import unpack
from .qr_utils import try_decode_qr


# ── Constants ────────────────────────────────────────────────────

# Max pixel dimension for QR detection. Frames larger than this are
# downscaled before detection to avoid wasting CPU on 4K+ input.
_MAX_DETECT_DIM = 1080

# Maximum frames the reader thread may prefetch ahead of the worker pool.
_READER_QUEUE_CAPACITY = 64

# ── crash-isolation dispatch hook ────────────────────────────────
# Worker functions call ``_dispatch_detect`` instead of
# ``try_decode_qr`` directly. :func:`decoder.extract_qr_from_video`
# swaps this to :meth:`qr_sandbox.SandboxedDetector.detect` when
# ``detect_isolation != 'off'`` and restores it on exit.


def _in_process_detect(_frame_idx: int, frame: "np.ndarray") -> str | None:
    return try_decode_qr(frame)


_dispatch_detect = _in_process_detect


# ── Frame reading ────────────────────────────────────────────────


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


def _read_frames(video_path, sample_rate, total_frames, start_frame=0):
    """Generator that reads frames from video.

    Yields ``(frame_idx, frame_ndarray)`` tuples. Frames are
    downscaled to ``_MAX_DETECT_DIM`` and passed as BGR uint8
    ndarrays directly.

    Thread-safety note: ``cv2.VideoCapture.read()`` reuses an
    internal frame buffer — we force a contiguous copy before
    yielding so each worker owns its frame outright.
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

    Yields ``(frame_idx, frame_ndarray)`` tuples.
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


# ── QR payload parsing ───────────────────────────────────────────


def _qr_text_to_block_and_seed(qr_data: str):
    """Parse a detected QR string into ``(block_bytes, seed)``.

    Tries the three payload encodings used historically by
    ``qrstream``: base45, base64, COBS/latin-1.

    Returns ``(block_bytes, seed)`` on success, ``(None, None)``
    otherwise.
    """
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
            return (candidate, header.seed)
        except (ValueError, struct.error):
            continue

    return (None, None)


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


# ── Detection workers ────────────────────────────────────────────


def _worker_detect_qr(frame_data, qr_detector=None):
    """Worker function for thread-pool QR detection.

    Takes ``(frame_idx, frame_ndarray)`` and an optional
    :class:`~qrstream.detector.base.QRDetector`.  When ``qr_detector``
    is given, detection routes through its ``detect()`` method.

    Returns ``(frame_idx, block_bytes_or_None, seed_or_None)``.
    """
    frame_idx, frame = frame_data
    if frame is None:
        return (frame_idx, None, None)

    if qr_detector is not None:
        qr_data = try_decode_qr(frame, qr_detector=qr_detector)
    else:
        qr_data = _dispatch_detect(frame_idx, frame)

    if qr_data is None:
        return (frame_idx, None, None)

    block_bytes, seed = _qr_text_to_block_and_seed(qr_data)
    return (frame_idx, block_bytes, seed)


def _worker_detect_qr_clahe(frame_data, qr_detector=None):
    """Recovery worker: run QR detect on a CLAHE-boosted frame copy.

    Used by ``_targeted_recovery`` after the main scan failed to
    deliver enough unique seeds for LT peeling to converge.

    Takes ``(frame_idx, frame_ndarray)`` and an optional detector.
    Returns ``(frame_idx, block_bytes_or_None, seed_or_None)``.
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

    if qr_detector is not None:
        qr_data = try_decode_qr(boosted, qr_detector=qr_detector)
    else:
        qr_data = _dispatch_detect(frame_idx, boosted)
    if qr_data is None:
        return (frame_idx, None, None)

    block_bytes, seed = _qr_text_to_block_and_seed(qr_data)
    return (frame_idx, block_bytes, seed)


# ── Prefetch iterator ────────────────────────────────────────────


def _prefetch_iter(source_iter, capacity: int = _READER_QUEUE_CAPACITY):
    """Run ``source_iter`` in a background thread, yielding items in order.

    This lets frame read + downscale overlap with worker-pool
    detection on the main thread.
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
        stop_event.set()
        while t.is_alive():
            try:
                item = q.get(timeout=0.1)
                if item is _SENTINEL:
                    break
            except Exception:
                break


# ── Pipelined stream scan ────────────────────────────────────────


def _stream_scan(executor: Executor, frame_iter, seen_seeds, unique_blocks,
                 decoded_count, no_detect_count, lt_decoder, pbar, verbose,
                 seed_frame_map, workers, worker_fn=None):
    """Pipelined scan: keep ``workers*2`` detect tasks in flight at all times.

    Reads frames via ``_prefetch_iter`` (background thread) and feeds
    them to ``executor`` using a sliding window of pending futures.
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

    # On early termination, cancel anything still queued.
    for fut in pending:
        fut.cancel()

    return decoded_count, no_detect_count, early_done


# ── Range utilities ──────────────────────────────────────────────


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
