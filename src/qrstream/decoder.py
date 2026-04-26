"""
LT Fountain Code Decoder: QR video → LT decode → file reconstruction.

Supports V2/V3 protocols with CRC32 validation.
Features adaptive sample rate and targeted frame recovery.

This module is the public API facade.  Implementation is split across:

- :mod:`_lt_decoder`  — ``LTDecoder``, ``decode_blocks``, ``decode_blocks_to_file``
- :mod:`_video_io`    — frame I/O, workers, payload parsing, stream scan
- :mod:`_probe`       — adaptive sample-rate probing
- :mod:`_recovery`    — targeted recovery for missing seeds

All public symbols are re-exported here so that existing
``from qrstream.decoder import X`` statements continue to work.
"""

import os
import struct
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import cv2
from tqdm import tqdm

from . import qr_sandbox
from .qr_utils import DETECTOR_CAN_CRASH  # noqa: F401 — re-export

# ── Re-exports from sub-modules ─────────────────────────────────
# Keep every symbol that was previously importable from decoder.py.

from ._lt_decoder import (  # noqa: F401
    LTDecoder,
    decode_blocks,
    decode_blocks_to_file,
    _decode_into_decoder,
)
from ._video_io import (  # noqa: F401
    _dispatch_detect,
    _in_process_detect,
    _downscale_frame,
    _qr_text_to_block_and_seed,
    _worker_detect_qr,
    _worker_detect_qr_clahe,
    _read_frames,
    _read_frame_ranges,
    _stream_scan,
    _prefetch_iter,
    _merge_ranges,
    _READER_QUEUE_CAPACITY,
    _MAX_DETECT_DIM,
)
from ._probe import (  # noqa: F401
    _build_probe_ranges,
    _compute_auto_sample_rate,
    _analyze_probe_window,
    _probe_sample_rate,
)
from ._recovery import (  # noqa: F401
    _estimate_frame_for_seed,
    _targeted_recovery,
)

# ── Isolation mode validation ────────────────────────────────────


def _validate_isolation_mode(mode: str) -> None:
    if mode not in ("on", "off"):
        raise ValueError(
            f"detect_isolation must be 'on' or 'off', got {mode!r}"
        )


# ── Public API ───────────────────────────────────────────────────


def extract_qr_from_video(video_path: str, sample_rate: int = 0,
                           verbose: bool = False, workers: int | None = None,
                           use_mnn: bool = False,
                           *, detect_isolation: str = "on",
                           decode_attempts: int = 1,
                           mnn_confidence_threshold: float = 0.0):
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
        use_mnn: When True, use MNN-accelerated detection with
            automatic fallback to OpenCV WeChatQRCode.  Default False
            preserves the existing behaviour.
        detect_isolation: ``'on'`` (default) runs QR detection in a pool
            of subprocess helpers so a native crash in
            ``cv2.wechat_qrcode_WeChatQRCode`` degrades to a single
            dropped frame. ``'off'`` runs detection in-process.
        decode_attempts: Number of zxing-cpp binarizer strategies to
            try per crop on the MNN path (ignored when ``use_mnn=False``).
            ``1`` (default): fastest, covers ~70% hit rate with zxing's
            built-in try_invert / try_rotate / try_downscale; validated
            to produce byte-identical payloads vs the 4-attempt variant
            on real-phone recordings.  ``2`` / ``3`` trade latency for
            ~0.3 pp extra crop-level coverage each.  See
            ``dev/wechatqrcode-mnn-poc/results/m3_report.md``.
        mnn_confidence_threshold: SSD detector confidence floor in
            ``[0.0, 1.0)`` for the MNN path (ignored when
            ``use_mnn=False``).  Default ``0.0`` keeps every positive
            detection.  Setting ``0.95`` typically saves 3-5 % wall
            clock on real-phone captures by skipping zxing-cpp work
            on bboxes that empirically never decode.  See
            ``results/m3_confidence_report.md`` and the
            ``QRSTREAM_MNN_CONFIDENCE_THRESHOLD`` env var.

    Returns a list of raw block byte strings.
    """
    import qrstream._video_io as _vio

    _validate_isolation_mode(detect_isolation)
    if decode_attempts not in (1, 2, 3):
        raise ValueError(
            f"decode_attempts must be 1, 2, or 3, got {decode_attempts!r}"
        )
    if not (isinstance(mnn_confidence_threshold, (int, float))
            and not isinstance(mnn_confidence_threshold, bool)
            and 0.0 <= float(mnn_confidence_threshold) < 1.0):
        raise ValueError(
            f"mnn_confidence_threshold must be a float in [0.0, 1.0), "
            f"got {mnn_confidence_threshold!r}"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0
    cap.release()

    if workers is None:
        workers = os.cpu_count() or 1

    # ── Detector router (MNN opt-in) ──────────────────────────────
    qr_router = None
    if use_mnn:
        try:
            from .detector import DetectorRouter
            qr_router = DetectorRouter(
                use_mnn=True,
                decode_attempts=decode_attempts,
                mnn_confidence_threshold=mnn_confidence_threshold,
            )
            if verbose:
                print(
                    f"MNN detector enabled: {qr_router.name} "
                    f"(decode_attempts={decode_attempts}, "
                    f"confidence_threshold={mnn_confidence_threshold})"
                )
        except Exception as e:
            if verbose:
                print(f"MNN detector init failed ({e}), using OpenCV fallback")
            qr_router = None

    if verbose:
        print(f"Video: {total_frames} frames, {src_fps:.1f} FPS, {duration:.1f}s")
        print(f"Using {workers} worker processes")

    sandbox = None
    original_dispatch = _vio._dispatch_detect
    sandbox_needed = (detect_isolation == "on") and (qr_router is None)
    if sandbox_needed:
        try:
            sandbox = qr_sandbox.SandboxedDetector(pool_size=3)
            _vio._dispatch_detect = sandbox.detect
        except Exception as exc:
            print(
                f"[sandbox] failed to initialise ({exc}); "
                f"falling back to in-process detection."
            )
            sandbox = None

    try:
        seen_seeds = set()
        unique_blocks = []
        decoded_count = 0
        no_detect_count = 0
        lt_decoder = LTDecoder()
        seed_frame_map: dict[int, int] = {}

        # ── Auto sample_rate probe ────────────────────────────────
        probe_results = []
        probe_count = 0
        leading_frames_probed = 0
        detect_rate = 1.0
        avg_repeat = 1.0

        if sample_rate <= 0:
            (auto_rate, probe_results, probe_count,
             leading_frames_probed, detect_rate, avg_repeat) = _probe_sample_rate(
                video_path, workers, verbose, qr_detector=qr_router)
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
                                if qr_router is not None and verbose:
                                    _print_router_stats(qr_router)
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
            remaining = total_frames - (last_reported + 1)
            if remaining > 0:
                pbar.update(remaining)

        main_worker = partial(_worker_detect_qr, qr_detector=qr_router)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            decoded_count, no_detect_count, early_done = _stream_scan(
                executor, _tracking_frame_iter(),
                seen_seeds, unique_blocks,
                decoded_count, no_detect_count, lt_decoder, pbar, verbose,
                seed_frame_map, workers, worker_fn=main_worker)
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
        if (not early_done and lt_decoder.initialized
                and not lt_decoder.done):
            unique_blocks, decoded_count, no_detect_count = _targeted_recovery(
                video_path, total_frames, src_fps, workers,
                seen_seeds, unique_blocks, decoded_count, no_detect_count,
                lt_decoder, avg_repeat, verbose, seed_frame_map,
                qr_detector=qr_router)

        if qr_router is not None and verbose:
            _print_router_stats(qr_router)

        return unique_blocks
    finally:
        _vio._dispatch_detect = original_dispatch
        if sandbox is not None:
            crashes = sandbox.crash_count
            sandbox.close()
            if crashes > 0:
                print(
                    f"[sandbox] detector crashed {crashes} time(s) "
                    f"during decode; affected frames treated as "
                    f"no-detect. Decoding proceeded normally."
                )


# ── Diagnostics ──────────────────────────────────────────────────


def _print_router_stats(router) -> None:
    """Emit DetectorRouter stats on the main console after extraction."""
    try:
        stats = router.get_stats()
    except Exception:
        return
    mnn_attempts = stats.get("mnn_attempts", 0)
    mnn_success = stats.get("mnn_success", 0)
    mnn_fallbacks = stats.get("mnn_fallbacks", 0)
    opencv_attempts = stats.get("opencv_attempts", 0)
    opencv_success = stats.get("opencv_success", 0)
    opencv_rescues = stats.get("opencv_rescues", 0)
    adaptive_disables = stats.get("adaptive_disables", 0)
    adaptive_enables = stats.get("adaptive_enables", 0)
    tail = ""
    if adaptive_disables or adaptive_enables:
        tail = (
            f", adaptive=[off×{adaptive_disables},"
            f"on×{adaptive_enables}]"
        )
    print(
        f"Detector stats: mnn={mnn_success}/{mnn_attempts} "
        f"(fallback={mnn_fallbacks}), "
        f"opencv={opencv_success}/{opencv_attempts} "
        f"(rescues={opencv_rescues}){tail}"
    )
