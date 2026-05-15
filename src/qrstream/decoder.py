"""
Fountain Code Decoder: QR video -> decode -> file reconstruction.

Supports V3 (LT fountain codes) and V4 (RaptorQ) protocols with
CRC32 validation.  Features adaptive sample rate and targeted frame
recovery.
"""

import io
import os
import struct
import time
import zlib
import base64
from collections import namedtuple
from math import ceil, log
from queue import Queue
from threading import Event, Thread
from concurrent.futures import (
    Executor,
    ThreadPoolExecutor,
    as_completed,
    FIRST_COMPLETED,
    wait as _futures_wait,
)

from ._compat import suppress_native_stderr

with suppress_native_stderr():
    import cv2
    import numpy as np
    import av

# Suppress verbose FFmpeg log output (info/warning level).
av.logging.set_level(av.logging.FATAL)

from .lt_codec import PRNG, BlockGraph, DEFAULT_C, DEFAULT_DELTA
from .protocol import unpack, V4_VERSION
from .raptorq_codec import RaptorQDecoder
from .qr_utils import try_decode_qr, try_decode_qr_with_bbox
from . import protocol as _protocol_mod
from .ui import ProgressReporter, QuietReporter, SlidingHitWindow


# Capacity tables for ``_infer_qr_modules``, keyed by version,
# materialised at import time.  EC=M (level 1) matches the qrstream
# encoder default.  range(1, 41) is already ascending, so no sort.
_QR_CAP_BYTE_M: list[tuple[int, int]] = [
    (v, _protocol_mod._QR_CAPACITY[(v, 1)]) for v in range(1, 41)
]
_QR_CAP_ALPHA_M: list[tuple[int, int]] = [
    (v, _protocol_mod._QR_CAPACITY_ALPHANUMERIC[(v, 1)]) for v in range(1, 41)
]


# ── Probe observation record ──────────────────────────────────────
# Named tuple for the 10-field result from ``_worker_probe_detect``,
# replacing positional indexing with readable field names throughout
# the probe and adaptive-downscale pipeline.
ProbeObservation = namedtuple("ProbeObservation", [
    "frame_idx",
    "block_bytes",
    "seed",
    "text_len",
    "is_alpha",
    "bbox_side",
    "frame_h",
    "frame_w",
    "bbox_cx",
    "bbox_cy",
])


class _ReaderFailure(RuntimeError):
    """Raised when a background frame reader fails."""


def _open_video_container(video_path: str):
    """Open a video via PyAV and normalise file-not-found behaviour."""
    try:
        return av.open(video_path)
    except FileNotFoundError:
        raise
    except Exception as exc:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Cannot open video file: {video_path}") from exc
        raise


# ── Reader / progress constants ──────────────────────────────────
# Maximum frames the reader thread may prefetch ahead of the worker
# pool.  Kept small to avoid memory bloat when detection is idle.
_READER_QUEUE_CAPACITY = 64

# ── Probe window parameters ──────────────────────────────────────
# Size of each contiguous probe window (frames).
_PROBE_WINDOW_SIZE = 120
# Gap between probe window centres as a fraction of the timeline.
_PROBE_GAP_RATIO = 0.15
# Per-seed detection probability target for sample_rate computation.
_TARGET_DETECT_PROB = 0.95

# ── Three-phase probe parameters ─────────────────────────────────
# Phase 1: crop exploration — consecutive frames per probe window.
_PROBE_CROP_BURST = 7
# Phase 2: PPM sweep — consecutive frames per probe window.
_PROBE_PPM_BURST = 5
# Phase 3 reader queue capacity for pipelined read+detect.
_PROBE_PIPELINE_QUEUE = 32

# ── CLAHE preprocessing ─────────────────────────────────────────
_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID_SIZE = (8, 8)


class LTDecoder:
    """Consumes LT fountain-coded V3 blocks and reconstructs the original data.

    Accepts V3 blocks with CRC validation; corrupt blocks are silently
    discarded.  For V4 (RaptorQ) blocks, use
    :class:`qrstream.raptorq_codec.RaptorQDecoder` instead.
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
        """Write the reconstructed output directly to a file.

        ``show_progress`` is kept for backward compatibility but is
        ignored — the Save phase is instantaneous in practice and the
        new UI layer surfaces it via ``reporter.save_done`` emitted
        by the caller.
        """
        del show_progress  # unused; signature preserved for API compat
        written = 0
        with open(output_path, 'wb') as f:
            if self.compressed:
                decompressor = zlib.decompressobj()
                try:
                    for chunk in self._iter_recovered_chunks():
                        data = decompressor.decompress(chunk)
                        if data:
                            f.write(data)
                            written += len(data)
                    tail = decompressor.flush()
                except zlib.error as e:
                    raise RuntimeError(
                        f"Decompression failed: {e}. Decoded payload may be corrupted.") from e
                if tail:
                    f.write(tail)
                    written += len(tail)
            else:
                for chunk in self._iter_recovered_chunks():
                    f.write(chunk)
                    written += len(chunk)
        return written


def _attempt_ge_checkpoint(lt_decoder,
                           reporter: ProgressReporter,
                           stage: str) -> bool:
    """Try a phase-boundary Gauss-Jordan rescue pass.

    Scan and targeted-recovery phases use cheap LT peeling for live
    progress.  At phase boundaries, a stalled peeling graph may already
    span all unknown source blocks; this checkpoint lets extraction stop
    early instead of scanning the next recovery level unnecessarily.

    No-op for RaptorQ decoders — they have no peeling graph and handle
    recovery internally.
    """
    if not lt_decoder.initialized or lt_decoder.done:
        return lt_decoder.done

    # RaptorQ has no Gauss-Jordan rescue path — skip entirely to avoid
    # emitting misleading GE progress events.
    if isinstance(lt_decoder, RaptorQDecoder):
        return lt_decoder.done

    reporter.ge_start(
        stage=stage,
        recovered=lt_decoder.num_recovered,
        k=lt_decoder.K,
    )
    try:
        rescued = lt_decoder.try_gaussian_rescue()
    finally:
        reporter.ge_done(
            success=lt_decoder.done,
            recovered=lt_decoder.num_recovered,
            k=lt_decoder.K,
        )

    return rescued


def _format_extraction_result(unique_blocks, lt_decoder,
                              return_decoder: bool):
    if not return_decoder:
        return unique_blocks
    completed_decoder = lt_decoder if lt_decoder.is_done() else None
    return unique_blocks, completed_decoder


# ── Video QR extraction (thread pool) ────────────────────────────

# Default fallback for the per-call max-dim (used when the adaptive
# probe has no usable bbox observations — e.g. probe failed to decode
# any frame, or the WeChat detector returned a degenerate bbox).  The
# previous behaviour was to *always* downscale to 1080; that wrecks
# detection on high-resolution camera captures of high-version QRs
# (V25+) where each module already only spans ~3-5 pixels.  We keep
# 1080 as the conservative fallback for legacy callers but the active
# decode path computes a per-video value via
# :func:`_compute_adaptive_max_dim` based on observed module density.
_MAX_DETECT_DIM = 1080

# Probe phase always runs at the source resolution (clamped to this
# generous cap) so :func:`_compute_adaptive_max_dim` sees the QR
# modules at full fidelity and can measure their pixel size
# accurately.  4320 covers up to 8K source frames; anything larger is
# downscaled to keep WeChat's preprocessing under control.
_PROBE_MAX_DETECT_DIM = 4320

# Fallback target ppm when the multi-resolution sweep cannot learn a
# per-video threshold (too few samples or plateau too low).  6.0 is
# conservative — it achieves plateau hit rate on all tested real-world
# captures (IMG_9432.MOV, v073 fixtures).  Synthetic / clean videos
# will learn a lower threshold and downscale more aggressively.
_ADAPTIVE_TARGET_PPM_FALLBACK = 6.0
# Never downscale below this dimension regardless of the adaptive
# computation; smaller frames invite WeChat false-negatives even on
# clean captures.
_ADAPTIVE_MIN_DETECT_DIM = 720
# Cap for the adaptive value so we don't run detection at 8K when a
# single probe frame happens to contain a tiny QR (which would imply
# the QR is unreadable anyway).
_ADAPTIVE_MAX_DETECT_DIM = 4320

# ── PPM-threshold learning parameters ────────────────────────────
# Minimum number of (post_ppm, decoded?) samples before the sliding-
# window fit is considered reliable.  Fewer samples → fall back to
# _ADAPTIVE_TARGET_PPM_FALLBACK.
_PPM_LEARN_MIN_SAMPLES = 30
# Width of the sliding window used to estimate hit_rate(ppm).
_PPM_LEARN_WINDOW_SIZE = 15
# The inflection point is the last sliding window below the midpoint
# of the transition band (floor → plateau), scaled by this fraction.
# 0.5 corresponds to the S-curve's 50%-of-range crossing, which is
# the most noise-robust estimator of the transition centre.  The
# safety margin (_PPM_LEARN_SAFETY_MARGIN) then pads upward.
_PPM_LEARN_PLATEAU_FRAC = 0.5
# Safety margin applied to the learned ppm threshold to account for
# frame-to-frame variation not captured by the probe sample.
_PPM_LEARN_SAFETY_MARGIN = 1.15
# Candidate downscale fractions for the multi-resolution sweep.
# Each is applied to the probe-measured src_max to produce a candidate
# max_dim.  The sweep runs from full-res (1.0) down to ~35% of the
# source; this range typically spans the ppm cliff.
_PPM_SWEEP_FRACTIONS = (1.0, 0.7, 0.5, 0.35)
# Reduced fractions for Phase 2 sweep (frac=1.0 reused from Phase 1).
_PPM_SWEEP_FRACTIONS_REDUCED = (0.7, 0.5, 0.35)
# Subsample ratio: use every Nth full-res probe frame for the sweep
# to keep the cost manageable (~60 frames out of 360).
_PPM_SWEEP_SUBSAMPLE = 6

# Recovery escalation: L2 boosts the main-scan max_dim by this factor.
# L3 always uses full source resolution (src_max).
_RECOVERY_ESCALATION_BOOST = 1.5

# ── Crop ROI parameters ──────────────────────────────────────────
# Stability threshold for the bbox-derived crop ROI.  If the normalised
# IQR of the QR centre position or apparent side length exceeds this
# fraction, the QR moves too much and cropping is disabled.
_CROP_STABILITY_THRESHOLD = 0.15
# Continuous margin range.  A perfectly stable QR uses the minimum;
# jitter approaching _CROP_STABILITY_THRESHOLD approaches the maximum.
_CROP_MARGIN_MIN = 1.15
_CROP_MARGIN_MAX = 1.65
# Legacy constant kept for API compatibility in tests; new code uses
# the continuous margins above.
_CROP_MARGIN = 1.3
# Minimum number of probe observations to derive a crop box.
_CROP_MIN_OBSERVATIONS = 10


def _crop_margin_for_jitter(norm_iqr: float) -> float:
    """Map normalised jitter continuously to a crop margin multiplier."""
    if norm_iqr <= 0.0:
        return _CROP_MARGIN_MIN
    t = min(1.0, norm_iqr / _CROP_STABILITY_THRESHOLD)
    return _CROP_MARGIN_MIN + (_CROP_MARGIN_MAX - _CROP_MARGIN_MIN) * t


def _downscale_frame(
    frame: np.ndarray, max_dim: int = _MAX_DETECT_DIM
) -> np.ndarray:
    """Downscale a frame if its larger dimension exceeds ``max_dim``."""
    h, w = frame.shape[:2]
    src_max = max(h, w)
    if src_max <= max_dim:
        return frame
    scale = max_dim / src_max
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _bbox_side_pixels(bbox: np.ndarray) -> float:
    """Estimate the QR side length in pixels from a (4, 2) corner array.

    WeChatQRCode returns the four QR corners in clockwise order; we
    average the four edge lengths so a slightly perspective-warped
    quadrilateral still yields a sensible scalar.  Returns ``0.0`` for
    degenerate / missing bboxes (zero-area).
    """
    if bbox is None or bbox.shape != (4, 2):
        return 0.0
    pts = bbox.astype(np.float64)
    edges = [
        pts[(i + 1) % 4] - pts[i]
        for i in range(4)
    ]
    lengths = [float(np.linalg.norm(e)) for e in edges]
    if min(lengths) <= 1.0:
        return 0.0
    return float(np.mean(lengths))


def _crop_frame(
    frame: np.ndarray,
    crop_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Crop ``frame`` to ``(y0, y1, x0, x1)`` if a crop box is given.

    Returns the original frame when ``crop_box`` is ``None``.  The
    returned array is a view (no copy) — callers downstream already
    force a ``.copy()`` before yielding to worker threads.
    """
    if crop_box is None:
        return frame
    y0, y1, x0, x1 = crop_box
    return frame[y0:y1, x0:x1]


def _prepare_frame(
    frame: np.ndarray,
    crop_box: tuple[int, int, int, int] | None,
    max_dim: int,
) -> np.ndarray:
    """Crop, downscale, and copy a frame for thread-safe detection.

    Applies the full ``crop → downscale → contiguous copy`` pipeline
    used by :func:`_read_frames` and :func:`_read_frame_ranges`.
    """
    frame = _crop_frame(frame, crop_box)
    frame = _downscale_frame(frame, max_dim=max_dim)
    return np.ascontiguousarray(frame)


def _derive_crop_box(
    probe_raw: list[tuple],
    frame_h: int,
    frame_w: int,
) -> tuple[int, int, int, int] | None:
    """Derive a stable crop ROI from probe-phase bbox observations.

    Uses a continuous adaptive margin based on QR centre/size jitter:

    - ``norm_iqr = 0`` → ``_CROP_MARGIN_MIN`` (tightest stable crop).
    - ``0 < norm_iqr ≤ _CROP_STABILITY_THRESHOLD`` → linearly
      interpolated margin up to ``_CROP_MARGIN_MAX``.
    - ``norm_iqr > _CROP_STABILITY_THRESHOLD`` → disabled (returns None).

    Returns ``None`` (= don't crop) when:
    - fewer than ``_CROP_MIN_OBSERVATIONS`` usable bbox observations,
    - the QR centre varies too much (see above).

    ``probe_raw`` entries are :class:`ProbeObservation` namedtuples from
    ``_worker_probe_detect``.
    """
    cxs: list[float] = []
    cys: list[float] = []
    sides: list[float] = []

    for entry in probe_raw:
        bbox_side = entry.bbox_side
        bbox_cx = entry.bbox_cx
        bbox_cy = entry.bbox_cy
        if bbox_side <= 0.0 or bbox_cx <= 0.0 or bbox_cy <= 0.0:
            continue
        cxs.append(bbox_cx)
        cys.append(bbox_cy)
        sides.append(bbox_side)

    if len(cxs) < _CROP_MIN_OBSERVATIONS:
        return None

    # Stability check: IQR of centre positions and QR apparent size.
    # Handheld captures may keep the QR centre almost fixed while the
    # camera moves closer/farther, so side-length jitter must also feed
    # the tight-vs-wide decision.
    cxs_sorted = sorted(cxs)
    cys_sorted = sorted(cys)
    sides_sorted = sorted(sides)
    q1_idx = len(cxs_sorted) // 4
    q3_idx = 3 * len(cxs_sorted) // 4
    iqr_x = cxs_sorted[q3_idx] - cxs_sorted[q1_idx]
    iqr_y = cys_sorted[q3_idx] - cys_sorted[q1_idx]
    iqr_side = sides_sorted[q3_idx] - sides_sorted[q1_idx]
    med_side = sides_sorted[len(sides_sorted) // 2]

    # Normalised IQR: max of X/Y relative to frame dim, and side IQR
    # relative to median QR side.
    norm_iqr_x = iqr_x / frame_w if frame_w > 0 else 1.0
    norm_iqr_y = iqr_y / frame_h if frame_h > 0 else 1.0
    norm_iqr_side = iqr_side / med_side if med_side > 0 else 1.0
    norm_iqr = max(norm_iqr_x, norm_iqr_y, norm_iqr_side)

    if norm_iqr > _CROP_STABILITY_THRESHOLD:
        return None

    # Continuous adaptive margin based on jitter magnitude.
    margin = _crop_margin_for_jitter(norm_iqr)

    # Median centre.
    med_cx = cxs_sorted[len(cxs_sorted) // 2]
    med_cy = cys_sorted[len(cys_sorted) // 2]

    # Build crop rect.
    half = int(med_side * margin / 2)
    cx_i, cy_i = int(med_cx), int(med_cy)

    x0 = max(0, cx_i - half)
    x1 = min(frame_w, cx_i + half)
    y0 = max(0, cy_i - half)
    y1 = min(frame_h, cy_i + half)

    # Sanity: crop must be meaningful (at least half the bbox side).
    if (x1 - x0) < med_side * 0.5 or (y1 - y0) < med_side * 0.5:
        return None

    return (y0, y1, x0, x1)


def _infer_qr_modules(text_len: int, alphanumeric: bool) -> int | None:
    """Reverse-lookup QR ``modules_per_side`` from a decoded payload.

    Picks the smallest ISO 18004 QR version whose capacity (at EC=M,
    the qrstream default) accommodates ``text_len`` characters of
    payload in the given mode.  Returns ``modules_per_side`` =
    ``4 * version + 17``, or ``None`` when the payload exceeds even
    V40 capacity (which would imply the source video was produced by
    a non-qrstream encoder we should not try to second-guess).

    EC level is fixed at M (the encoder default); using the wrong EC
    level only over-estimates the version by 1 step in the worst case,
    which biases the adaptive max-dim slightly larger (more pixels per
    module) — that errs on the side of preserving information, which
    is the whole point of this heuristic.
    """
    if text_len <= 0:
        return None
    if alphanumeric:
        table = _QR_CAP_ALPHA_M
    else:
        table = _QR_CAP_BYTE_M
    for version, cap in table:
        if text_len <= cap:
            return 4 * version + 17
    return None


def _compute_adaptive_max_dim(
    frame_h: int,
    frame_w: int,
    bbox: np.ndarray,
    modules_per_side: int | None,
    target_ppm: float = _ADAPTIVE_TARGET_PPM_FALLBACK,
) -> int | None:
    """Compute a per-video downscale target preserving QR module density.

    Given a probe frame's resolution, the QR's bounding box on that
    frame, and (optionally) the inferred modules_per_side, return the
    largest-dimension cap we can apply during the main scan while
    still leaving each module ≥ ``target_ppm`` pixels.

    Returns ``None`` when the inputs are insufficient to compute a
    sensible value (caller should fall back to ``_MAX_DETECT_DIM``).
    """
    qr_side = _bbox_side_pixels(bbox)
    if qr_side <= 0.0 or modules_per_side is None or modules_per_side <= 0:
        return None

    src_max = max(frame_h, frame_w)
    if src_max <= 0:
        return None

    pixels_per_module = qr_side / modules_per_side
    if pixels_per_module <= target_ppm:
        # Source already at-or-below the target density; keep the
        # frame at its native resolution (clamped to the upper cap).
        return min(src_max, _ADAPTIVE_MAX_DETECT_DIM)

    scale = target_ppm / pixels_per_module
    target = int(src_max * scale)
    target = max(_ADAPTIVE_MIN_DETECT_DIM, target)
    target = min(_ADAPTIVE_MAX_DETECT_DIM, target)
    # Never go *above* the source — there's no point upscaling.
    target = min(src_max, target)
    return target


# ── PPM-threshold learning ───────────────────────────────────────

def _learn_ppm_threshold(
    samples: list[tuple[float, bool]],
) -> float | None:
    """Learn the minimum post-downscale ppm that sustains detection.

    ``samples`` is a list of ``(post_ppm, decoded)`` observations
    collected by :func:`_multi_res_sweep`.  The function fits a
    hit-rate curve via a sliding window over ppm-sorted samples and
    finds the inflection point — the smallest ppm at which the hit
    rate reaches ``_PPM_LEARN_PLATEAU_FRAC`` of its global plateau.

    Returns the learned ``target_ppm`` (with a
    ``_PPM_LEARN_SAFETY_MARGIN`` applied), or ``None`` when the
    input is too sparse or the plateau is too low (< 30% hit rate)
    to be meaningful.
    """
    if len(samples) < _PPM_LEARN_MIN_SAMPLES:
        return None

    # Sort by ppm ascending.
    sorted_s = sorted(samples, key=lambda x: x[0])
    w = _PPM_LEARN_WINDOW_SIZE

    if len(sorted_s) < w:
        return None

    # Build (median_ppm, hit_rate) per window position.
    windows: list[tuple[float, float]] = []
    for i in range(len(sorted_s) - w + 1):
        chunk = sorted_s[i : i + w]
        median_ppm = chunk[w // 2][0]
        hit_rate = sum(1 for _, d in chunk if d) / w
        windows.append((median_ppm, hit_rate))

    if not windows:
        return None

    plateau = max(rate for _, rate in windows)
    if plateau < 0.3:
        # Detector is unreliable on this video even at full
        # resolution — fall back to the caller's default.
        return None

    # Find the floor (hit rate at the low-ppm end).
    floor = min(rate for _, rate in windows[:max(1, len(windows) // 4)])

    # Midpoint of the transition band.
    midpoint = floor + _PPM_LEARN_PLATEAU_FRAC * (plateau - floor)

    # Walk the windows from left (low ppm) to right (high ppm).
    # The inflection is the *last* window below the midpoint —
    # the ppm just above it is where detection stabilises.
    inflection_ppm = windows[0][0]
    for ppm, rate in windows:
        if rate < midpoint:
            inflection_ppm = ppm
        else:
            break

    learned = inflection_ppm * _PPM_LEARN_SAFETY_MARGIN
    return round(learned, 2)


def _multi_res_sweep(
    probe_frames: list[tuple[int, np.ndarray]],
    src_max: int,
    qr_side_full: float,
    modules: int,
    workers: int,
    *,
    fractions: tuple[float, ...] = _PPM_SWEEP_FRACTIONS,
    subsample: int = _PPM_SWEEP_SUBSAMPLE,
) -> list[tuple[float, bool]]:
    """Re-detect a subset of probe frames at multiple resolutions.

    For each candidate downscale fraction in ``fractions``,
    downscale the frames and run QR detection.  For each frame × resolution
    combination, record ``(post_ppm, detected)`` where ``post_ppm``
    is the predicted ppm at the candidate resolution.

    Returns the collected samples, sorted by ppm ascending.  The
    caller feeds these into :func:`_learn_ppm_threshold`.
    """
    if not probe_frames or src_max <= 0 or modules <= 0 or qr_side_full <= 0:
        return []

    # K = qr_side_full / (src_max × modules)  — video-specific constant
    K = qr_side_full / (src_max * modules)

    # Subsample frames to keep sweep cost manageable.
    subset = probe_frames[::subsample] if subsample > 1 else probe_frames
    if not subset:
        return []

    # Build work items: (frame_idx, downscaled_frame, candidate_ppm)
    work_items: list[tuple[int, np.ndarray, float]] = []
    for frac in fractions:
        cand_max = max(_ADAPTIVE_MIN_DETECT_DIM, int(src_max * frac))
        cand_max = min(cand_max, src_max)
        cand_ppm = cand_max * K
        for frame_idx, frame in subset:
            scaled = _downscale_frame(frame, max_dim=cand_max)
            work_items.append((frame_idx, scaled, cand_ppm))

    # Dispatch all detections in parallel.
    samples: list[tuple[float, bool]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_ppm = {}
        for frame_idx, scaled_frame, cand_ppm in work_items:
            fut = executor.submit(
                try_decode_qr, scaled_frame
            )
            future_to_ppm[fut] = cand_ppm

        for fut in as_completed(future_to_ppm):
            ppm = future_to_ppm[fut]
            try:
                result = fut.result()
                samples.append((ppm, result is not None))
            except Exception:
                samples.append((ppm, False))

    samples.sort(key=lambda x: x[0])
    return samples


def _worker_detect_qr(frame_data):
    """Worker function for thread-pool QR detection.

    Takes (frame_idx, frame_ndarray).
    Returns (frame_idx, block_bytes_or_None, seed_or_None).

    The frame is a ``numpy.ndarray`` (BGR uint8, already downscaled
    to ``_MAX_DETECT_DIM``) handed to the worker by reference: under
    ``ThreadPoolExecutor`` workers share the main process address
    space, so the ndarray travels as a zero-copy reference.
    zxing-cpp is reentrant and safe to call from multiple threads.
    """
    frame_idx, frame = frame_data
    if frame is None:
        return (frame_idx, None, None)

    qr_data = try_decode_qr(frame)
    if qr_data is None:
        return (frame_idx, None, None)

    result = _try_decode_qr_payload(qr_data)
    if result is not None:
        block_bytes, seed, _ = result
        return (frame_idx, block_bytes, seed)

    return (frame_idx, None, None)


def _worker_detect_qr_clahe(frame_data):
    """Recovery worker: run the detector on a CLAHE-boosted copy of the frame.

    Used by ``_targeted_recovery`` after the main scan failed to
    deliver enough unique seeds for LT peeling to converge.  CLAHE
    (Contrast Limited Adaptive Histogram Equalisation) is a purely
    scalar, per-tile operation that boosts local contrast on QR modules,
    lifting frames that were just below the detector threshold back
    above it.

    Takes ``(frame_idx, frame_ndarray)``. Returns
    ``(frame_idx, block_bytes_or_None, seed_or_None)``.
    """
    frame_idx, frame = frame_data
    if frame is None:
        return (frame_idx, None, None)

    try:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]
        clahe = cv2.createCLAHE(
            clipLimit=_CLAHE_CLIP_LIMIT,
            tileGridSize=_CLAHE_TILE_GRID_SIZE,
        )
        ycrcb[:, :, 0] = clahe.apply(y)
        boosted = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    except cv2.error:
        return (frame_idx, None, None)

    if not boosted.flags['C_CONTIGUOUS']:
        boosted = np.ascontiguousarray(boosted)

    qr_data = try_decode_qr(boosted)
    if qr_data is None:
        return (frame_idx, None, None)

    result = _try_decode_qr_payload(qr_data)
    if result is not None:
        block_bytes, seed, _ = result
        return (frame_idx, block_bytes, seed)

    return (frame_idx, None, None)


def _worker_probe_detect(frame_data):
    """Probe-phase worker: detect QR + return bbox + decoded mode tag.

    Returns a :class:`ProbeObservation` namedtuple with fields:
    ``frame_idx, block_bytes, seed, text_len, is_alpha,
    bbox_side, frame_h, frame_w, bbox_cx, bbox_cy``.

    The trailing fields feed :func:`_compute_adaptive_max_dim`
    (module density), :func:`_derive_crop_box` (centre stability),
    and :func:`_multi_res_sweep` (PPM threshold learning).
    """
    frame_idx, frame = frame_data
    if frame is None:
        return ProbeObservation(frame_idx, None, None, 0, False,
                                0.0, 0, 0, 0.0, 0.0)

    h, w = frame.shape[:2]

    detected = try_decode_qr_with_bbox(frame)
    if detected is None:
        return ProbeObservation(frame_idx, None, None, 0, False,
                                0.0, h, w, 0.0, 0.0)
    qr_text, bbox = detected

    bbox_side = _bbox_side_pixels(bbox)
    text_len = len(qr_text)

    # Compute bbox centre from the 4 corner points.
    if bbox is not None and bbox.shape == (4, 2):
        bbox_cx = float(bbox[:, 0].mean())
        bbox_cy = float(bbox[:, 1].mean())
    else:
        bbox_cx, bbox_cy = 0.0, 0.0

    result = _try_decode_qr_payload(qr_text)
    if result is not None:
        block_bytes, seed, is_alpha = result
        return ProbeObservation(frame_idx, block_bytes, seed,
                                text_len, is_alpha, bbox_side, h, w,
                                bbox_cx, bbox_cy)

    # Detected but undecodable — still surface the bbox so we can
    # estimate density even when the payload doesn't validate.
    return ProbeObservation(frame_idx, None, None, text_len, True,
                            bbox_side, h, w, bbox_cx, bbox_cy)


def _try_decode_qr_payload(
    qr_data: str,
) -> tuple[bytes, int, bool] | None:
    """Try all QR-payload decode strategies and return the first success.

    Strategies (tried in order):
      1) base45 (current default for high-density / alphanumeric mode)
      2) base64 (standard byte mode)
      3) COBS/latin-1 (legacy pre-0.6 high-density mode)

    Returns ``(block_bytes, seed, is_alphanumeric)`` on success, or
    ``None`` when no strategy produces a valid protocol block.
    """
    from .protocol import base45_decode, cobs_decode

    strategies = (
        (True,  lambda d: _try_base45(d, base45_decode)),
        (False, _try_base64),
        (False, lambda d: _try_cobs(d, cobs_decode)),
    )
    for is_alpha, decode_fn in strategies:
        candidate = decode_fn(qr_data)
        if candidate is None:
            continue
        try:
            header, _ = unpack(candidate)
            return (candidate, header.seed, is_alpha)
        except (ValueError, struct.error):
            continue
    return None


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


# ── Video metadata helper ─────────────────────────────────────────

def _get_video_info(video_path: str) -> tuple[int, float, int, int]:
    """Return (total_frames, fps, width, height) from a video file.

    Uses a multi-tier fallback for total_frames because some container
    formats (MKV, WebM, certain MOV) do not store nb_frames metadata
    and PyAV reports ``stream.frames == 0`` even though the file has
    decodable content.

    Fallback tiers:
      1. ``stream.frames`` — direct metadata field.
      2. ``stream.duration * stream.time_base * fps`` — stream-level duration.
      3. ``container.duration / AV_TIME_BASE * fps`` — container-level duration.
    """
    container = _open_video_container(video_path)
    try:
        stream = container.streams.video[0]

        fps = float(stream.average_rate) if stream.average_rate else 0.0
        width = stream.width or 0
        height = stream.height or 0

        # Tier 1: direct stream metadata (nb_frames).
        total_frames = stream.frames if stream.frames and stream.frames > 0 else 0

        # Tier 2: derive from stream duration + time_base + fps.
        if total_frames <= 0 and stream.duration and stream.time_base and fps > 0:
            duration_sec = float(stream.duration * stream.time_base)
            total_frames = int(duration_sec * fps)

        # Tier 3: derive from container-level duration (microseconds).
        if total_frames <= 0 and container.duration and fps > 0:
            duration_sec = container.duration / 1_000_000.0
            total_frames = int(duration_sec * fps)

        return total_frames, fps, width, height
    finally:
        container.close()


def _read_frames(video_path, sample_rate, total_frames, start_frame=0,
                 max_detect_dim: int = _MAX_DETECT_DIM,
                 crop_box: tuple[int, int, int, int] | None = None):
    """Generator that reads frames from video using PyAV.

    Yields ``(frame_idx, frame_ndarray)`` tuples. Frames are optionally
    cropped to ``crop_box`` (y0, y1, x0, x1), then downscaled to
    ``max_detect_dim``.

    PyAV decodes directly to numpy via ``frame.to_ndarray()``, so
    frames are independent and thread-safe without extra copying.
    """
    container = _open_video_container(video_path)
    stream = container.streams.video[0]
    # Do not use stream.thread_type = "AUTO": FFmpeg frame-level threading
    # deadlocks with Python generators when the internal decoded-frame
    # queue fills while the generator is suspended at yield.  Decoding is
    # already overlapped with detection via _prefetch_iter's producer thread.
    frame_idx = 0
    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_idx < start_frame:
                    frame_idx += 1
                    continue
                if (frame_idx - start_frame) % sample_rate == 0:
                    img = frame.to_ndarray(format="bgr24")
                    img = _prepare_frame(img, crop_box, max_detect_dim)
                    yield (frame_idx, img)
                frame_idx += 1
    finally:
        container.close()


def _read_frame_ranges(video_path, frame_ranges,
                       max_detect_dim: int = _MAX_DETECT_DIM,
                       crop_box: tuple[int, int, int, int] | None = None):
    """Generator that reads specific frame ranges from video using PyAV.

    Seeks independently to each range for efficiency on long videos
    where ranges may be widely spaced.

    Args:
        frame_ranges: list of (start_frame, end_frame) tuples (inclusive).
        max_detect_dim: per-call downscale cap (see :func:`_read_frames`).
        crop_box: optional (y0, y1, x0, x1) ROI applied before downscale.
    """
    if not frame_ranges:
        return

    sorted_ranges = sorted(frame_ranges)

    container = _open_video_container(video_path)
    stream = container.streams.video[0]
    # Do not use stream.thread_type = "AUTO": frame-level threading
    # deadlocks with Python generators (see _read_frames comment).

    try:
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        if not fps or not stream.time_base:
            return

        for range_start, range_end in sorted_ranges:
            # Seek to the start of each range independently.
            target_pts = int(range_start / fps / float(stream.time_base))
            container.seek(target_pts, stream=stream)

            for packet in container.demux(stream):
                done_range = False
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    frame_idx = round(
                        float(frame.pts * stream.time_base) * fps
                    )
                    if frame_idx > range_end:
                        done_range = True
                        break
                    if frame_idx >= range_start:
                        img = frame.to_ndarray(format="bgr24")
                        img = _prepare_frame(img, crop_box, max_detect_dim)
                        yield (frame_idx, img)
                if done_range:
                    break
    finally:
        container.close()


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


def _build_phase_burst_ranges(
    total_frames: int, burst: int, gap_ratio: float = _PROBE_GAP_RATIO,
    offset: int = 0,
) -> list[tuple[int, int]]:
    """Build short burst ranges at probe window centres for Phase 1/2.

    Returns 3 ranges of ``burst`` consecutive frames, positioned at the
    same centres as :func:`_build_probe_ranges` but much shorter.
    ``offset`` shifts the starting position within each window so
    Phase 1 and Phase 2 sample different frames from the same windows.
    """
    if total_frames <= 0 or burst <= 0:
        return []
    if total_frames <= burst:
        return [(0, total_frames - 1)]

    centers = [0.5 - gap_ratio, 0.5, 0.5 + gap_ratio]
    ranges = []
    for ratio in centers:
        ratio = min(max(ratio, 0.0), 1.0)
        center = int(round((total_frames - 1) * ratio))
        start = max(0, center - burst // 2 + offset)
        end = start + burst - 1
        if end >= total_frames:
            end = total_frames - 1
            start = max(0, end - burst + 1)
        ranges.append((start, end))

    return _merge_ranges(ranges)


def _compute_auto_sample_rate(detect_rate: float, avg_repeat: float,
                              total_frames: int = 0,
                              K_estimate: int = 0) -> int:
    """Compute a conservative sample rate from one probe window.

    When ``K_estimate`` and ``total_frames`` are provided, clamps the
    result so the expected number of unique seeds collected at the
    given ``detect_rate`` exceeds ``K_estimate × 1.5`` (the minimum
    for reliable LT convergence).
    """
    p = detect_rate

    if p >= 0.99:
        rate = max(1, int(avg_repeat / 1.5))
    elif p > 0.01:
        min_chances = log(1 - _TARGET_DETECT_PROB) / log(1 - p)
        rate = max(1, int(avg_repeat / min_chances))
    else:
        rate = 1

    # Conservative clamp: ensure enough frames are scanned to collect
    # K × 1.5 unique seeds at the observed detect rate.
    if K_estimate > 0 and total_frames > 0 and p > 0.01:
        min_unique_needed = int(K_estimate * 1.5)
        # Each sampled frame yields ~p unique seeds (simplified; actual
        # dedup lowers this, but overestimating frames needed is safe).
        min_frames_needed = min_unique_needed / p
        max_rate = max(1, int(total_frames / min_frames_needed))
        if rate > max_rate:
            rate = max_rate

    return rate


def _analyze_probe_window(window_results, total_frames: int = 0,
                          K_estimate: int = 0):
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
        sample_rate = _compute_auto_sample_rate(
            detect_rate, avg_repeat,
            total_frames=total_frames,
            K_estimate=K_estimate,
        )

    return {
        'frame_count': frame_count,
        'detect_rate': detect_rate,
        'avg_repeat': avg_repeat,
        'distinct_seed_count': len(distinct_seeds),
        'sample_rate': sample_rate,
    }


def _probe_sample_rate(video_path: str, workers: int,
                       verbose: bool = False,
                       reporter: ProgressReporter | None = None):
    """Probe multiple windows of a video to determine optimal sample_rate.

    Uses a three-phase pipeline to minimise detection cost:

      Phase 1 — Crop exploration: detect QR on a small number of
        full-resolution frames to derive a crop ROI and video constants.
      Phase 2 — Resolution exploration: run a multi-resolution PPM
        sweep on cropped frames to learn the adaptive downscale cap.
      Phase 3 — Sample rate estimation: detect QR on the full probe
        window set using crop + adaptive resolution in a pipelined
        read-detect mode, then compute sample_rate from detection
        statistics.

    Returns:
        (sample_rate, probe_results, probe_count, leading_frames_probed,
         detect_rate, avg_repeat, adaptive_max_dim, crop_box)

        ``adaptive_max_dim`` is ``None`` when the probe gathered no
        usable bbox observations; callers should fall back to
        ``_MAX_DETECT_DIM`` in that case.
        ``crop_box`` is ``None`` when the QR centre is unstable or
        insufficient data is available.
    """
    if reporter is None:
        reporter = QuietReporter()

    total_frames, _, _, _ = _get_video_info(video_path)

    if total_frames <= 0:
        return 1, [], 0, 0, 0.0, 1.0, None, None

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Crop Exploration
    # ═══════════════════════════════════════════════════════════════
    # Read a small burst of frames at full resolution from each probe
    # window centre to derive crop_box and video constants.  Uses the
    # same window centres as the full probe to ensure spatial coverage.
    reporter.probe_update(scanned=0, total=0, detect=0.0, phase="crop")

    phase1_ranges = _build_phase_burst_ranges(
        total_frames, _PROBE_CROP_BURST, _PROBE_GAP_RATIO, offset=0,
    )
    phase1_frames = list(_read_frame_ranges(
        video_path, phase1_ranges, max_detect_dim=_PROBE_MAX_DETECT_DIM,
    ))

    if not phase1_frames:
        return 1, [], 0, 0, 0.0, 1.0, None, None

    # Detect on Phase 1 frames (full resolution, no crop).
    phase1_raw: list = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker_probe_detect, fd): fd[0]
                   for fd in phase1_frames}
        for future in as_completed(futures):
            phase1_raw.append(future.result())
    phase1_raw.sort(key=lambda x: x.frame_idx)

    # Derive crop box from Phase 1 bbox observations.
    probe_fh, probe_fw = 0, 0
    for entry in phase1_raw:
        if entry.frame_h > 0 and entry.frame_w > 0:
            probe_fh, probe_fw = entry.frame_h, entry.frame_w
            break

    crop_box = _derive_crop_box(phase1_raw, probe_fh, probe_fw)

    # Extract video-level constants (src_max, qr_side, modules).
    _probe_obs = _extract_probe_video_constants(phase1_raw)

    if verbose:
        phase1_detect_count = sum(1 for o in phase1_raw if o.seed is not None)
        reporter.debug(
            f"Phase 1 (crop): {len(phase1_raw)} frames, "
            f"detected={phase1_detect_count}/{len(phase1_raw)}"
        )
        if crop_box is not None:
            y0, y1, x0, x1 = crop_box
            reporter.debug(
                f"Crop ROI: ({x0},{y0})-({x1},{y1}) "
                f"= {x1-x0}x{y1-y0}px "
                f"(frame {probe_fw}x{probe_fh})"
            )
        elif probe_fh > 0:
            reporter.debug(
                "Crop ROI: disabled (unstable QR position or insufficient data)"
            )

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Resolution / PPM Exploration
    # ═══════════════════════════════════════════════════════════════
    reporter.probe_update(scanned=0, total=0, detect=0.0, phase="calibrating")

    learned_ppm: float | None = None

    if _probe_obs is not None:
        src_max_obs, qr_side_obs, modules_obs = _probe_obs

        # Read a separate burst of frames WITH crop applied.
        phase2_ranges = _build_phase_burst_ranges(
            total_frames, _PROBE_PPM_BURST, _PROBE_GAP_RATIO,
            offset=_PROBE_CROP_BURST,  # offset to avoid overlapping Phase 1
        )
        phase2_frames = list(_read_frame_ranges(
            video_path, phase2_ranges,
            max_detect_dim=_PROBE_MAX_DETECT_DIM,
            crop_box=crop_box,
        ))

        # Determine the effective src_max for the sweep — the max dim
        # of the frames actually passed to _multi_res_sweep.
        if crop_box is not None and phase2_frames:
            _fh2, _fw2 = phase2_frames[0][1].shape[:2]
            sweep_src_max = max(_fh2, _fw2)
        else:
            sweep_src_max = src_max_obs

        # Run multi-resolution sweep at reduced fractions (skip 1.0,
        # Phase 1 already provides full-resolution detection data).
        sweep_samples = _multi_res_sweep(
            phase2_frames, sweep_src_max, qr_side_obs, modules_obs,
            workers,
            fractions=_PPM_SWEEP_FRACTIONS_REDUCED,
            subsample=1,  # use all Phase 2 frames (already small set)
        )

        # Inject Phase 1 full-resolution observations as frac=1.0 data.
        # PPM at full resolution: qr_side_obs / modules_obs.
        full_ppm = qr_side_obs / modules_obs if modules_obs > 0 else 0.0
        if full_ppm > 0:
            for entry in phase1_raw:
                detected = entry.seed is not None or entry.bbox_side > 0
                sweep_samples.append((full_ppm, detected))

        sweep_samples.sort(key=lambda x: x[0])
        learned_ppm = _learn_ppm_threshold(sweep_samples)

        if verbose:
            if learned_ppm is not None:
                reporter.debug(
                    f"Phase 2 (PPM): {len(sweep_samples)} samples → "
                    f"learned_target_ppm={learned_ppm:.2f}"
                )
            else:
                reporter.debug(
                    f"Phase 2 (PPM): {len(sweep_samples)} samples → "
                    f"insufficient data, "
                    f"fallback={_ADAPTIVE_TARGET_PPM_FALLBACK}"
                )

    adaptive_max_dim = _adaptive_max_dim_from_probe(phase1_raw, learned_ppm)
    effective_ppm = learned_ppm if learned_ppm is not None else _ADAPTIVE_TARGET_PPM_FALLBACK
    if verbose and adaptive_max_dim is not None:
        reporter.debug(
            f"Adaptive downscale: max_detect_dim={adaptive_max_dim}px "
            f"(target_ppm={effective_ppm:.2f}, "
            f"default would be {_MAX_DETECT_DIM}px)"
        )

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Sample Rate Exploration (pipelined read + detect)
    # ═══════════════════════════════════════════════════════════════
    # Use crop + adaptive resolution for bulk detection.
    effective_max_dim = (
        adaptive_max_dim if adaptive_max_dim is not None else _MAX_DETECT_DIM
    )

    probe_ranges = _build_probe_ranges(
        total_frames, _PROBE_WINDOW_SIZE, _PROBE_GAP_RATIO,
    )
    _expected_read = sum(
        max(0, end - start + 1) for start, end in probe_ranges
    )

    if verbose and len(probe_ranges) > 1:
        ranges_str = ", ".join(f"{start}-{end}" for start, end in probe_ranges)
        reporter.debug(f"Phase 3 probe windows: {ranges_str}")

    # Pipelined execution: reader thread feeds a bounded queue,
    # detector threads consume from it concurrently.
    _QUEUE_SENTINEL = object()
    frame_queue: Queue = Queue(maxsize=_PROBE_PIPELINE_QUEUE)
    read_done = Event()
    probe_raw: list = []
    _probe_detected = 0
    _probe_lock = __import__('threading').Lock()
    reader_error: list[BaseException] = []

    def _reader():
        try:
            for fd in _read_frame_ranges(
                    video_path, probe_ranges,
                    max_detect_dim=effective_max_dim,
                    crop_box=crop_box):
                frame_queue.put(fd)
        except BaseException as exc:
            reader_error.append(exc)
        finally:
            read_done.set()
            frame_queue.put(_QUEUE_SENTINEL)

    reader_thread = Thread(target=_reader, daemon=True)
    reader_thread.start()

    # Consume frames and submit detection work.
    probe_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending_futures = {}
        frames_submitted = 0

        while True:
            # Try to fill the executor with work.
            while len(pending_futures) < workers * 2:
                try:
                    fd = frame_queue.get(timeout=0.05)
                except Exception:
                    if read_done.is_set() and frame_queue.empty():
                        break
                    continue
                if fd is _QUEUE_SENTINEL:
                    break
                fut = executor.submit(_worker_probe_detect, fd)
                pending_futures[fut] = fd[0]
                frames_submitted += 1
                break  # submit one, then check for completions

            # Harvest completed futures.
            if pending_futures:
                done_set, _ = _futures_wait(
                    pending_futures, timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                for fut in done_set:
                    result = fut.result()
                    with _probe_lock:
                        probe_raw.append(result)
                        if result.seed is not None:
                            _probe_detected += 1
                    del pending_futures[fut]
                    reporter.probe_update(
                        scanned=len(probe_raw),
                        total=max(_expected_read, len(probe_raw)),
                        detect=(_probe_detected / len(probe_raw)
                                if probe_raw else 0.0),
                        phase="scanning",
                    )

            # Exit when reader is done and all work completed.
            if (read_done.is_set() and frame_queue.empty()
                    and not pending_futures):
                break

    reader_thread.join()
    if reader_error:
        raise _ReaderFailure("probe frame reader failed") from reader_error[0]
    probe_count = len(probe_raw)
    # The probe touches disjoint windows across the timeline; the main
    # scan cannot skip a contiguous prefix based on that sparse sample.
    leading_frames_probed = 0

    if not probe_raw:
        return 1, [], 0, 0, 0.0, 1.0, adaptive_max_dim, crop_box

    # Sort by frame index.
    probe_raw.sort(key=lambda x: x.frame_idx)

    # Build probe_results from all probe phases.
    # Phase 3 remains authoritative for sample_rate because it uses the
    # final crop/downscale settings, but Phase 1 contributes sparse early
    # blocks so the main decode can start with every unique seed already
    # seen during probing.
    phase3_results = [
        (t.frame_idx, t.block_bytes, t.seed) for t in probe_raw
    ]
    phase3_frame_set = {t.frame_idx for t in probe_raw}

    # Collect Phase 1 decoded blocks not already in Phase 3.
    extra_results = [
        (t.frame_idx, t.block_bytes, t.seed)
        for t in phase1_raw
        if t.block_bytes is not None and t.frame_idx not in phase3_frame_set
    ]
    probe_results = extra_results + phase3_results

    # ── Extract K estimate from probe-decoded blocks ────────────
    K_estimate = 0
    for _, block_bytes, seed in probe_results:
        if block_bytes is not None:
            try:
                hdr, _ = unpack(block_bytes, skip_crc=True)
                K_estimate = hdr.block_count
                break
            except (ValueError, struct.error):
                continue

    window_stats = []
    for start, end in probe_ranges:
        window_results = [result for result in phase3_results
                          if start <= result[0] <= end]
        stats = _analyze_probe_window(
            window_results,
            total_frames=total_frames,
            K_estimate=K_estimate,
        )
        window_stats.append((start, end, stats))

    valid_windows = [entry for entry in window_stats if entry[2]['sample_rate'] is not None]
    if not valid_windows:
        reporter.debug(
            f"Probe: {probe_count} frames, insufficient seed diversity "
            f"→ sample_rate=1"
        )
        return (1, probe_results, probe_count, leading_frames_probed,
                0.0, 1.0, adaptive_max_dim, crop_box)

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
            reporter.debug(
                f"  Probe window {start}-{end}: "
                f"detect_rate={stats['detect_rate']:.0%}, "
                f"avg_repeat={stats['avg_repeat']:.1f}, "
                f"seeds={stats['distinct_seed_count']}, "
                f"sample_rate={rate_str}"
            )
        reporter.debug(
            f"Probe: {probe_count} frames across {len(probe_ranges)} "
            f"windows, limiting_window={limiting_start}-{limiting_end}, "
            f"detect_rate={detect_rate:.0%}, avg_repeat={avg_run:.1f} "
            f"→ sample_rate={auto_rate}"
        )

    return (auto_rate, probe_results, probe_count,
            leading_frames_probed, detect_rate, avg_run,
            adaptive_max_dim, crop_box)


def _extract_probe_video_constants(
    probe_raw: list[tuple],
) -> tuple[int, float, int] | None:
    """Extract (src_max, median_qr_side, modules) from probe observations.

    Returns ``None`` when no usable bbox+modules observations exist.
    """
    obs: list[tuple[int, float, int]] = []
    for entry in probe_raw:
        if entry.bbox_side <= 0.0:
            continue
        if entry.frame_h <= 0 or entry.frame_w <= 0:
            continue
        if entry.text_len <= 0:
            continue
        modules = _infer_qr_modules(entry.text_len, entry.is_alpha)
        if modules is None:
            continue
        obs.append((max(entry.frame_h, entry.frame_w),
                    entry.bbox_side, modules))

    if not obs:
        return None

    # Use median for stability.
    src_maxs = sorted(o[0] for o in obs)
    sides = sorted(o[1] for o in obs)
    # modules should be identical across frames (same QR version).
    modules_val = obs[0][2]
    return (
        src_maxs[len(src_maxs) // 2],
        sides[len(sides) // 2],
        modules_val,
    )


def _adaptive_max_dim_from_probe(
    probe_raw: list[tuple],
    learned_ppm: float | None = None,
) -> int | None:
    """Aggregate probe-worker observations into a single max-detect-dim.

    ``probe_raw`` items are :class:`ProbeObservation` namedtuples emitted
    by :func:`_worker_probe_detect`.

    When ``learned_ppm`` is provided (from :func:`_learn_ppm_threshold`),
    it replaces the static fallback constant to set the target module
    density.  Otherwise ``_ADAPTIVE_TARGET_PPM_FALLBACK`` is used.

    Strategy: take the *median* of the per-frame max-dim suggestions
    so single-frame outliers (motion blur warping the bbox, edge
    captures) don't dominate.
    """
    target_ppm = (
        learned_ppm if learned_ppm is not None
        else _ADAPTIVE_TARGET_PPM_FALLBACK
    )

    suggestions: list[int] = []
    for entry in probe_raw:
        if entry.bbox_side <= 0.0:
            continue
        if entry.frame_h <= 0 or entry.frame_w <= 0:
            continue
        if entry.text_len <= 0:
            continue
        modules = _infer_qr_modules(entry.text_len, entry.is_alpha)
        if modules is None:
            continue
        synth_bbox = np.array(
            [[0.0, 0.0], [entry.bbox_side, 0.0],
             [entry.bbox_side, entry.bbox_side], [0.0, entry.bbox_side]],
            dtype=np.float32,
        )
        target = _compute_adaptive_max_dim(
            entry.frame_h, entry.frame_w, synth_bbox, modules,
            target_ppm=target_ppm,
        )
        if target is not None:
            suggestions.append(target)

    if not suggestions:
        return None
    suggestions.sort()
    mid = suggestions[len(suggestions) // 2]
    return int(mid)


def _get_eliminated(decoder) -> dict:
    """Return the block-recovery dict for either decoder type.

    * ``LTDecoder`` → ``block_graph.eliminated``  (block_idx → numpy data)
    * ``RaptorQDecoder`` → ``eliminated``          (block_idx → True)

    Both satisfy the ``block_idx in recovered`` membership test that
    :func:`qrstream.ui.compute_block_map_cells` requires.

    TODO: Optimize RaptorQ recovery block map.  Currently we only mark
    source ESIs as eliminated (known precisely) and fill repair coverage
    on decode completion.  Future: track repair symbol recovery incrementally
    during decoding for finer-grained progress UI.
    """
    if not decoder.initialized:
        return {}
    # RaptorQDecoder carries .eliminated directly.
    if isinstance(decoder, RaptorQDecoder):
        return decoder.eliminated
    # LTDecoder stores it inside the block graph.
    bg = getattr(decoder, 'block_graph', None)
    return bg.eliminated if bg is not None else {}


class _ScanProgressTracker:
    """Track main-scan progress and throttle reporter churn."""

    _EMIT_INTERVAL = 0.1

    def __init__(self, *, total_frames: int,
                 leading_frames_probed: int,
                 lt_decoder,
                 reporter: ProgressReporter):
        self.total_frames = total_frames
        self.processed_frames = leading_frames_probed
        self.lt_decoder = lt_decoder
        self.reporter = reporter
        self.hit_window = SlidingHitWindow(capacity=128)
        self.last_emit_ts = 0.0

    def mark_skipped_until(self, frame_idx: int) -> None:
        if frame_idx > self.processed_frames:
            self.processed_frames = frame_idx

    def _payload(self, *, video_pct: float | None = None) -> dict[str, object]:
        if video_pct is None:
            video_pct = (
                self.processed_frames / self.total_frames * 100
                if self.total_frames > 0 else 100.0
            )
        # LTDecoder: block_graph.eliminated (dict of block_idx → data)
        # RaptorQDecoder: eliminated (dict of block_idx → True)
        # Both support ``block_idx in recovered`` membership tests
        # required by compute_block_map_cells().
        recovered = _get_eliminated(self.lt_decoder)
        k = self.lt_decoder.K if self.lt_decoder.initialized else None
        return {
            "video_pct": video_pct,
            "hit_window": self.hit_window.ratio,
            "file_pct": self.lt_decoder.progress * 100,
            "recovered": recovered,
            "k": k,
        }

    def on_frame(self, frame_idx: int, hit: bool) -> None:
        if frame_idx + 1 > self.processed_frames:
            self.processed_frames = frame_idx + 1
        self.hit_window.push(hit)
        now = time.monotonic()
        if now - self.last_emit_ts < self._EMIT_INTERVAL:
            return
        self.last_emit_ts = now
        self.reporter.scan_update(**self._payload())

    def emit_final(self) -> None:
        try:
            self.reporter.scan_update(**self._payload(video_pct=100.0))
        except Exception:
            pass


def _tracked_read_frames(video_path: str, sample_rate: int,
                         total_frames: int,
                         *, start_frame: int,
                         max_detect_dim: int,
                         crop_box: tuple[int, int, int, int] | None,
                         scan_tracker: _ScanProgressTracker):
    """Wrap ``_read_frames`` while accounting for skipped frame positions."""
    for frame_data in _read_frames(
            video_path, sample_rate, total_frames,
            start_frame=start_frame,
            max_detect_dim=max_detect_dim,
            crop_box=crop_box):
        scan_tracker.mark_skipped_until(frame_data[0])
        yield frame_data


def extract_qr_from_video(video_path: str, sample_rate: int = 0,
                           verbose: bool = False, workers: int | None = None,
                           *,
                           reporter: ProgressReporter | None = None,
                           return_decoder: bool = False):
    """Extract unique QR code payloads from a video file.

    Uses an LT decoder internally for early termination: stops scanning
    as soon as all source blocks are recovered.

    When initial scan doesn't recover all blocks, performs targeted
    recovery by reading only the video segments corresponding to
    missing seeds.

    Args:
        sample_rate: Process every Nth frame. 0 = auto-detect (default).
        verbose: Emit verbose diagnostic details (routed through
            ``reporter.debug``).
        workers: Number of parallel worker threads.
        reporter: Optional :class:`qrstream.ui.ProgressReporter`.  When
            ``None`` a :class:`QuietReporter` is used (no progress
            output) so the function stays side-effect-free for
            programmatic callers.
        return_decoder: When true, return ``(blocks, decoder)`` where
            ``decoder`` is the completed extraction decoder when scan-phase
            peeling/GE already reconstructed the file; otherwise ``None``.

    Returns a list of raw block byte strings by default.
    """
    if reporter is None:
        reporter = QuietReporter()

    total_frames, src_fps, src_w, src_h = _get_video_info(video_path)
    src_max = max(src_w, src_h)
    duration = total_frames / src_fps if src_fps > 0 else 0

    if workers is None:
        workers = os.cpu_count() or 1

    if verbose:
        reporter.debug(
            f"Video: {total_frames} frames, {src_fps:.1f} FPS, {duration:.1f}s"
        )
        reporter.debug(f"Using {workers} worker threads")

    seen_seeds = set()
    unique_blocks = []
    decoded_count = 0
    no_detect_count = 0
    lt_decoder = None   # Auto-detect: LTDecoder or RaptorQDecoder
    seed_frame_map: dict[int, int] = {}  # observed seed → first frame index

    # ── Auto sample_rate probe ────────────────────────────────
    probe_results = []
    probe_count = 0
    leading_frames_probed = 0
    detect_rate = 1.0
    avg_repeat = 1.0
    adaptive_max_dim: int | None = None
    crop_box: tuple[int, int, int, int] | None = None

    if sample_rate <= 0:
        reporter.probe_start()
        (auto_rate, probe_results, probe_count,
         leading_frames_probed, detect_rate, avg_repeat,
         adaptive_max_dim, crop_box) = _probe_sample_rate(
            video_path, workers, verbose, reporter=reporter)
        sample_rate = auto_rate

        # crop_reduction = 1 - crop_area / frame_area
        crop_reduction: float | None
        if crop_box is not None and src_w > 0 and src_h > 0:
            y0, y1, x0, x1 = crop_box
            crop_area = max(0, (y1 - y0)) * max(0, (x1 - x0))
            frame_area = src_h * src_w
            if frame_area > 0:
                crop_reduction = max(
                    0.0, 1.0 - (crop_area / frame_area)
                )
            else:
                crop_reduction = None
        else:
            crop_reduction = None

        reporter.probe_done(
            sample=sample_rate,
            detect=detect_rate,
            repeat=avg_repeat,
            crop_reduction=crop_reduction,
            observed=probe_count,
            max_dim=adaptive_max_dim,
        )

        if verbose:
            reporter.debug(f"Using auto sample_rate={sample_rate}")

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
                        if lt_decoder is None:
                            lt_decoder, hdr, data = _create_decoder_for_block(
                                block_bytes)
                            done, _ = lt_decoder.consume_block(hdr, data)
                        else:
                            done, _ = lt_decoder.decode_bytes(block_bytes, skip_crc=True)
                        if done:
                            if verbose:
                                reporter.debug(
                                    f"Extraction done (during probe): "
                                    f"{probe_count} sampled frames, "
                                    f"{decoded_count} unique blocks"
                                )
                            return _format_extraction_result(
                                unique_blocks, lt_decoder, return_decoder)
                    except (ValueError, struct.error):
                        pass
            else:
                no_detect_count += 1

    # Fall back to the legacy global cap when no adaptive value
    # was derived (e.g. caller passed sample_rate>0 → no probe;
    # or probe failed to decode any QR).
    active_max_dim = (
        adaptive_max_dim if adaptive_max_dim is not None else _MAX_DETECT_DIM
    )

    if verbose and probe_count > 0:
        pct = lt_decoder.progress * 100 if lt_decoder is not None else 0.0
        reporter.debug(
            f"After probe: {decoded_count} unique blocks, "
            f"progress={pct:.1f}%"
        )

    # Ensure decoder exists for the main scan (fallback to LTDecoder
    # when the probe didn't decode any block).
    if lt_decoder is None:
        lt_decoder = LTDecoder()

    # ── Main scan (remaining frames) ─────────────────────────
    reporter.scan_start(total_frames=total_frames)
    scan_tracker = _ScanProgressTracker(
        total_frames=total_frames,
        leading_frames_probed=leading_frames_probed,
        lt_decoder=lt_decoder,
        reporter=reporter,
    )
    early_done = False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        decoded_count, no_detect_count, early_done, detect_count, lt_decoder = _stream_scan(
            executor,
            _tracked_read_frames(
                video_path,
                sample_rate,
                total_frames,
                start_frame=leading_frames_probed,
                max_detect_dim=active_max_dim,
                crop_box=crop_box,
                scan_tracker=scan_tracker,
            ),
            seen_seeds,
            unique_blocks,
            decoded_count,
            no_detect_count,
            lt_decoder,
            scan_tracker.on_frame,
            verbose,
            seed_frame_map,
            workers,
            reporter=reporter,
        )
        # Update the tracker's decoder reference in case _stream_scan
        # swapped it (V4 auto-detection).
        scan_tracker.lt_decoder = lt_decoder
        if early_done and verbose:
            reporter.debug(
                "Early termination: all source blocks recovered!"
            )

    scan_tracker.emit_final()
    reporter.scan_done()

    if (not early_done and lt_decoder.initialized
            and not lt_decoder.done):
        early_done = _attempt_ge_checkpoint(
            lt_decoder, reporter, "scan")

    total_sampled = detect_count + no_detect_count
    if verbose:
        hit_rate_str = (
            f"{detect_count * 100 // total_sampled}%"
            if total_sampled else "n/a"
        )
        status = " (early termination)" if early_done else ""
        reporter.debug(
            f"Extraction done{status}: {total_frames} frames "
            f"({total_sampled} sampled, sample_rate={sample_rate}, "
            f"hit={hit_rate_str}), "
            f"{decoded_count} unique blocks, "
            f"{no_detect_count} missed"
        )

    # ── Targeted recovery for missing seeds ───────────────────
    # Triggered whenever the main scan finished without the fountain
    # decoder converging, regardless of ``sample_rate`` or codec.
    # Recovery re-scans video segments where missing seeds are expected,
    # using CLAHE contrast boost and/or higher resolution to coax the
    # QR detector into reading frames it missed on the first pass.
    #
    # This is codec-agnostic: both LT and RaptorQ benefit from having
    # more unique QR frames extracted from the video.  The only
    # LT-specific logic *inside* recovery (GE checkpoints) is already
    # guarded by _attempt_ge_checkpoint() which no-ops for RaptorQ.
    if (not early_done and lt_decoder.initialized
            and not lt_decoder.done):
        unique_blocks, decoded_count, no_detect_count = _targeted_recovery(
            video_path, total_frames, src_fps, workers,
            seen_seeds, unique_blocks, decoded_count, no_detect_count,
            lt_decoder, avg_repeat, verbose, seed_frame_map,
            max_detect_dim=active_max_dim,
            src_max=src_max,
            crop_box=crop_box,
            reporter=reporter)

    return _format_extraction_result(
        unique_blocks, lt_decoder, return_decoder)


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
                       max_detect_dim: int = _MAX_DETECT_DIM,
                       src_max: int | None = None,
                       crop_box: tuple[int, int, int, int] | None = None,
                       reporter: ProgressReporter | None = None):
    """Multi-level recovery: escalate resolution on missing-seed frames.

    Runs up to three recovery levels with increasing ``max_detect_dim``
    to rescue frames the main scan missed due to insufficient ppm or
    contrast:

    - **L1** (same resolution + CLAHE): rescues contrast-bound frames.
    - **L2** (1.5× resolution): moderate ppm boost for near-miss frames.
    - **L3** (full source resolution): maximum ppm, last resort.

    Each level scans only the video segments where missing seeds are
    expected.  If the LT decoder converges after any level, remaining
    levels are skipped.

    Uses observed (seed, frame_idx) mapping from probe and main scan to
    build a linear model for estimating missing seed positions. Falls back
    to naive linear estimation when insufficient observations are available.
    """
    if reporter is None:
        reporter = QuietReporter()
    if seed_frame_map is None:
        seed_frame_map = {}

    if not seen_seeds:
        return unique_blocks, decoded_count, no_detect_count

    max_seed = max(seen_seeds)
    frames_per_qr = max(1, avg_repeat)

    all_seeds = set(range(1, max_seed + 1))
    missing_seeds = all_seeds - seen_seeds

    if not missing_seeds:
        return unique_blocks, decoded_count, no_detect_count

    # Build frame ranges for missing seeds.
    frame_ranges = []
    margin = max(2, int(frames_per_qr * 0.5))

    for seed in sorted(missing_seeds):
        center = _estimate_frame_for_seed(
            seed, seed_frame_map, frames_per_qr, total_frames)
        start = max(0, center - margin)
        end = min(total_frames - 1, center + int(frames_per_qr) + margin)
        frame_ranges.append((start, end))

    frame_ranges = _merge_ranges(frame_ranges)
    target_frames = sum(e - s + 1 for s, e in frame_ranges)

    # Determine effective src_max for escalation.
    if src_max is None or src_max <= 0:
        src_max = max_detect_dim  # no escalation possible

    # Build escalation levels, deduplicating identical max_dims.
    # L1/L2 use the probe-derived crop ROI; L3 (full resolution)
    # does NOT crop — it's the last resort and should use every
    # pixel available.
    l2_dim = min(src_max, int(max_detect_dim * _RECOVERY_ESCALATION_BOOST))
    levels: list[tuple[int, object, str, tuple | None]] = [
        (max_detect_dim, _worker_detect_qr_clahe, "L1-clahe", crop_box),
    ]
    if l2_dim > max_detect_dim:
        levels.append((l2_dim, _worker_detect_qr, "L2-boost", crop_box))
    if src_max > l2_dim:
        levels.append((src_max, _worker_detect_qr, "L3-full", None))

    if verbose:
        reporter.debug(
            f"Targeted recovery: {len(missing_seeds)} missing seeds, "
            f"{target_frames} frames in {len(frame_ranges)} segments, "
            f"{len(levels)} escalation level(s)"
        )

    _EMIT_INTERVAL = 0.1

    for level_dim, worker_fn, level_name, level_crop in levels:
        if lt_decoder.done:
            break

        # Re-evaluate missing seeds after previous levels.
        current_missing = all_seeds - seen_seeds
        if not current_missing:
            break

        # Rebuild frame ranges for currently-missing seeds only.
        level_ranges = []
        for seed in sorted(current_missing):
            center = _estimate_frame_for_seed(
                seed, seed_frame_map, frames_per_qr, total_frames)
            start = max(0, center - margin)
            end = min(total_frames - 1, center + int(frames_per_qr) + margin)
            level_ranges.append((start, end))
        level_ranges = _merge_ranges(level_ranges)
        level_frames = sum(e - s + 1 for s, e in level_ranges)

        if verbose:
            reporter.debug(
                f"{level_name}: max_dim={level_dim}, "
                f"{len(current_missing)} missing seeds, "
                f"{level_frames} frames"
            )

        reporter.recover_start(
            level=level_name,
            segments=level_ranges,
            total_frames=total_frames,
        )
        hit_window = SlidingHitWindow(capacity=128)
        rec_state = {
            "processed": 0,
            "last_emit_ts": 0.0,
            "current_range": level_ranges[0] if level_ranges else None,
        }

        def _recover_update(fidx: int, hit: bool) -> None:
            rec_state["processed"] += 1
            hit_window.push(hit)
            # Update which segment the current frame lives in (cheap
            # linear scan — level_ranges is small, usually < 20).
            for seg in level_ranges:
                if seg[0] <= fidx <= seg[1]:
                    rec_state["current_range"] = seg
                    break
            now = time.monotonic()
            if now - rec_state["last_emit_ts"] < _EMIT_INTERVAL:
                return
            rec_state["last_emit_ts"] = now
            progress_pct = (
                rec_state["processed"] / level_frames * 100
                if level_frames > 0 else 100.0
            )
            file_pct = lt_decoder.progress * 100
            recovered = _get_eliminated(lt_decoder)
            k = lt_decoder.K if lt_decoder.initialized else None
            reporter.recover_update(
                progress_pct=progress_pct,
                hit_window=hit_window.ratio,
                file_pct=file_pct,
                recovered=recovered,
                k=k,
                current_range=rec_state["current_range"],
            )

        level_decoded_before = decoded_count

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                decoded_count, no_detect_count, early_done, _, lt_decoder = _stream_scan(
                    executor,
                    _read_frame_ranges(video_path, level_ranges,
                                       max_detect_dim=level_dim,
                                       crop_box=level_crop),
                    seen_seeds, unique_blocks,
                    decoded_count, no_detect_count, lt_decoder,
                    _recover_update, verbose, seed_frame_map, workers,
                    worker_fn=worker_fn,
                    reporter=reporter)
                if early_done and verbose:
                    reporter.debug(f"{level_name}: all blocks recovered!")
        finally:
            # Final 100% tick before closing this level.
            try:
                reporter.recover_update(
                    progress_pct=100.0,
                    hit_window=hit_window.ratio,
                    file_pct=lt_decoder.progress * 100,
                    recovered=_get_eliminated(lt_decoder),
                    k=lt_decoder.K if lt_decoder.initialized else None,
                    current_range=rec_state["current_range"],
                )
            except Exception:
                pass
            reporter.recover_done()

        if early_done:
            break

        if (decoded_count > level_decoded_before
                and lt_decoder.initialized
                and not lt_decoder.done):
            if _attempt_ge_checkpoint(lt_decoder, reporter, level_name):
                break

    if verbose:
        status = " (complete)" if lt_decoder.done else ""
        final_missing = len(all_seeds - seen_seeds)
        tail = (f", {final_missing} seeds still missing"
                if final_missing > 0 else "")
        reporter.debug(
            f"Targeted recovery done{status}: "
            f"{decoded_count} unique blocks, {no_detect_count} missed{tail}"
        )

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
    _SENTINEL = object()
    q: Queue = Queue(maxsize=capacity)
    stop_event = Event()
    producer_error: list[BaseException] = []

    def _producer():
        try:
            for item in source_iter:
                if stop_event.is_set():
                    return
                q.put(item)
        except BaseException as exc:
            producer_error.append(exc)
        finally:
            q.put(_SENTINEL)

    t = Thread(target=_producer, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is _SENTINEL:
                if producer_error:
                    raise _ReaderFailure("prefetch reader failed") from producer_error[0]
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
                 decoded_count, no_detect_count, lt_decoder,
                 on_frame, verbose,
                 seed_frame_map, workers, worker_fn=None,
                 reporter: ProgressReporter | None = None):
    """Pipelined scan: keep ``workers*2`` detect tasks in flight at all times.

    Reads frames via ``_prefetch_iter`` (background thread) and feeds
    them to ``executor`` using a sliding window of pending futures.
    After each completed future, ``on_frame(frame_idx, hit_bool)`` is
    invoked --- the caller owns progress / hit-window rendering via a
    :class:`qrstream.ui.ProgressReporter`.

    ``lt_decoder`` may be an LTDecoder, RaptorQDecoder, or an
    uninitialised LTDecoder.  When the first successfully-decoded
    block is V4, the LTDecoder is transparently swapped out for a
    RaptorQDecoder.

    ``worker_fn`` defaults to :func:`_worker_detect_qr` (plain QR
    detection on the already-downscaled frame). Targeted recovery
    passes :func:`_worker_detect_qr_clahe` to rescue frames the main
    scan missed by the ε-margin introduced by cross-architecture
    ``cv2.resize(INTER_AREA)`` SIMD drift.
    """
    if worker_fn is None:
        worker_fn = _worker_detect_qr

    early_done = False
    IN_FLIGHT = max(workers * 2, 4)
    detect_count = 0  # ALL successful detections (unique + duplicate)

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
            hit = block_bytes is not None and seed is not None
            if hit:
                detect_count += 1
                if seed_frame_map is not None and seed not in seed_frame_map:
                    seed_frame_map[seed] = fidx
                if seed not in seen_seeds:
                    seen_seeds.add(seed)
                    unique_blocks.append(block_bytes)
                    decoded_count += 1
                    try:
                        if not lt_decoder.initialized:
                            # First valid block — auto-detect codec.
                            new_dec, hdr, data = _create_decoder_for_block(
                                block_bytes)
                            if type(new_dec) is not type(lt_decoder):
                                lt_decoder = new_dec
                            done, _ = lt_decoder.consume_block(hdr, data)
                        else:
                            done, _ = lt_decoder.decode_bytes(
                                block_bytes, skip_crc=True)
                        if done:
                            early_done = True
                    except (ValueError, struct.error):
                        pass
                    if verbose and reporter is not None:
                        pct = lt_decoder.progress * 100
                        reporter.debug(
                            f"Frame {fidx}: seed={seed}, "
                            f"uniq={decoded_count}, "
                            f"progress={pct:.1f}%"
                        )
            else:
                no_detect_count += 1

            if on_frame is not None:
                try:
                    on_frame(fidx, hit)
                except Exception:
                    pass

            # Keep the pool topped up — one in, one out.
            if not early_done:
                _submit_next()

    # On early termination, cancel anything still queued so we release
    # the executor promptly.
    for fut in pending:
        fut.cancel()

    # Explicitly close the prefetch generator so its producer thread
    # and the underlying PyAV container are cleaned up without waiting
    # for GC (which can be delayed indefinitely by thread references).
    prefetched.close()

    return decoded_count, no_detect_count, early_done, detect_count, lt_decoder


def _create_decoder_for_block(block_bytes: bytes):
    """Peek at the version byte and return the right decoder type.

    Returns ``(decoder, header, data)`` on success, or raises
    ``ValueError`` if the block cannot be parsed.
    """
    header, data = unpack(block_bytes, skip_crc=True)
    if header.version == V4_VERSION:
        decoder = RaptorQDecoder()
    else:
        decoder = LTDecoder()
    return decoder, header, data


def _decode_into_decoder(blocks, verbose=False,
                         reporter: ProgressReporter | None = None):
    """Feed blocks into an auto-detected decoder to reconstruct the file.

    Automatically selects LTDecoder (V3) or RaptorQDecoder (V4) based
    on the version byte of the first valid block.
    """
    if reporter is None:
        reporter = QuietReporter()
    if not blocks:
        # Business error: keep on stdout so ``capsys`` tests still see it.
        print("Error: No blocks to decode")
        return None

    decoder = None

    try:
        for i, block_bytes in enumerate(blocks):
            try:
                if decoder is None:
                    # Auto-detect codec from the first valid block.
                    decoder, header, data = _create_decoder_for_block(
                        block_bytes)
                    done, compressed = decoder.consume_block(header, data)
                else:
                    done, compressed = decoder.decode_bytes(block_bytes)
                if done:
                    if verbose:
                        reporter.debug(
                            f"Decoded after {i + 1}/{len(blocks)} blocks "
                            f"(filesize={decoder.filesize}, K={decoder.K}, "
                            f"compressed={compressed}, "
                            f"v={decoder.protocol_version})"
                        )
                    return decoder
            except ValueError as e:
                if verbose:
                    reporter.debug(f"Block {i} error, skipping: {e}")
            except Exception as e:
                if verbose:
                    reporter.debug(f"Block {i} error: {e}")
    finally:
        pass

    if decoder is None:
        print("Error: No valid blocks found")
        return None

    # Peeling (belief-propagation) exhausted all blocks without
    # converging. Attempt a GF(2) Gauss-Jordan rescue pass over the
    # accumulated check-node graph: if the surviving equations
    # collectively span the missing source blocks, we still get a
    # perfect reconstruction.  This path is only entered on peeling
    # failure, so it costs nothing in the healthy case.
    #
    # For RaptorQ decoders this is a no-op: recovery is handled
    # internally by the raptorq library and there is no peeling graph
    # to rescue.
    #
    # TODO(v0.10.0): the main reason peeling fails on a post-0.8
    # stream is legacy prng_version=0 encoding. Once v0 support is
    # dropped (see ``protocol.py``), revisit whether the rescue is
    # still worth carrying — native v1 streams converge above the
    # CLI's ``_MIN_OVERHEAD`` floor, so GE would only help
    # overhead-below-floor edge cases.
    if (decoder.initialized and not decoder.done
            and not isinstance(decoder, RaptorQDecoder)):
        rescued = decoder.try_gaussian_rescue()
        if rescued:
            if verbose:
                reporter.debug(
                    f"GE rescue recovered all "
                    f"{decoder.num_recovered}/{decoder.K} blocks "
                    f"after peeling stalled."
                )
            else:
                reporter.info(
                    f"GE rescue recovered "
                    f"{decoder.num_recovered}/{decoder.K} source blocks."
                )
            return decoder
        elif verbose:
            reporter.debug(
                f"GE rescue attempted, still "
                f"{decoder.num_recovered}/{decoder.K} recovered."
            )

    n_recovered = decoder.num_recovered
    k = decoder.K if decoder.K else '?'
    # Business failure messages: keep on stdout so ``capsys`` tests and
    # ``print``-based error capture still work.
    print(f"\nDecoding incomplete: {n_recovered}/{k} source blocks recovered "
          f"from {len(blocks)} encoded blocks.")
    print("Try recording the QR stream longer to capture more unique frames.")
    return None


def decode_blocks(blocks, verbose=False,
                  reporter: ProgressReporter | None = None) -> "bytes | None":
    """Feed blocks into an auto-detected decoder to reconstruct the file."""
    decoder = _decode_into_decoder(blocks, verbose=verbose, reporter=reporter)
    if decoder is None:
        return None
    try:
        return decoder.bytes_dump()
    except RuntimeError as e:
        print(f"Error: {e}")
        return None


def decode_blocks_to_file(blocks, output_path: str, verbose=False,
                          reporter: ProgressReporter | None = None,
                          decoder=None) -> "int | None":
    """Decode blocks and write the result directly to a file."""
    if reporter is None:
        reporter = QuietReporter()
    if decoder is not None and not decoder.is_done():
        decoder = None
    if decoder is None:
        decoder = _decode_into_decoder(blocks, verbose=verbose, reporter=reporter)
    if decoder is None:
        return None
    try:
        written = decoder.bytes_dump_to_file(output_path)
    except RuntimeError as e:
        print(f"Error: {e}")
        return None
    reporter.save_done(output_path=output_path, bytes_written=written)
    return written
