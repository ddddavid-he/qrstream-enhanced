"""
Detector router: selects and manages the active QR detector backend.

Responsibilities:
- Choose between OpenCV WeChatQRCode (default) and MNN based on
  the ``use_mnn`` flag.
- Auto-fallback: if the MNN detector fails to initialise or returns
  an error at runtime, transparently fall back to OpenCV.
- Expose a single ``detect(frame)`` call that hides backend selection.
- Future-proof: designed to integrate with ``--detect-isolation`` /
  ``DETECTOR_CAN_CRASH`` from the ``fix/wechat-native-crash`` branch.

Usage::

    router = DetectorRouter(use_mnn=True)
    result = router.detect(frame)
    if result.text:
        print(result.text, "via", result.backend)
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from pathlib import Path

import numpy as np

from .base import QRDetector, DetectResult
from .opencv_wechat import OpenCVWeChatDetector

logger = logging.getLogger(__name__)


class DetectorRouter(QRDetector):
    """Routes QR detection to the appropriate backend with fallback.

    The router itself implements ``QRDetector`` so it can be used
    anywhere a detector is expected — including the existing worker
    pool in ``decoder.py``.
    """

    # The router itself doesn't crash — it delegates to backends.
    # The ``DETECTOR_CAN_CRASH`` of the *active* backend determines
    # whether sandbox isolation is needed.
    DETECTOR_CAN_CRASH: bool = False

    def __init__(
        self,
        use_mnn: bool = False,
        mnn_model_dir: str | Path | None = None,
        mnn_backend: str | None = None,
        use_sr: bool = True,
        opencv_fallback: bool = True,
        *,
        adaptive_fallback: bool = True,
        adaptive_warmup: int = 64,
        adaptive_window: int = 256,
        adaptive_disable_rate: float = 0.02,
        adaptive_enable_rate: float = 0.05,
        adaptive_probe_interval: int = 64,
    ) -> None:
        self._use_mnn = use_mnn
        self._mnn_model_dir = mnn_model_dir
        self._mnn_backend = mnn_backend
        self._use_sr = use_sr
        # When True, a no-detect result from MNN triggers an OpenCV
        # re-scan on the same frame.  When False, MNN's verdict is
        # final — this is the low-latency mode for videos where
        # most frames contain a QR and OpenCV retry is pure overhead.
        self._opencv_fallback = opencv_fallback

        # ── Adaptive fallback controller ─────────────────────────
        # On videos where MNN already catches everything OpenCV can,
        # the default ``opencv_fallback=True`` path pays an extra
        # ~90ms/frame OpenCV call for every MNN miss and recovers
        # essentially nothing (IMG_9425: 677 OpenCV rescues, 0 extra
        # decodes with padding enabled).  We compute a rolling
        # "rescue rate" — the fraction of MNN misses that OpenCV
        # actually saves — and flip ``opencv_fallback`` to False
        # when that rate drops near zero.  Hysteresis (two separate
        # thresholds) prevents flapping when the true rescue rate
        # sits near the boundary.
        self._adaptive_fallback = adaptive_fallback and opencv_fallback
        self._adaptive_warmup = max(1, adaptive_warmup)
        self._adaptive_window = max(self._adaptive_warmup, adaptive_window)
        self._adaptive_disable_rate = adaptive_disable_rate
        self._adaptive_enable_rate = adaptive_enable_rate
        # While fallback is suppressed, run OpenCV once every
        # ``adaptive_probe_interval`` MNN misses to keep feeding the
        # rescue-rate window.  Without these probes the adaptive
        # switch would be a one-way door: once off, no rescue
        # observations could ever arrive to turn it back on.
        self._adaptive_probe_interval = max(1, adaptive_probe_interval)
        # Counter of MNN misses since the last probe while fallback
        # is suppressed.  Reset whenever a probe fires.
        self._suppressed_miss_count = 0
        # Rolling log of recent (mnn_miss, opencv_rescue) pairs.
        # Only entries where MNN missed are appended — success cases
        # don't influence the rescue-rate decision.
        self._rescue_log: deque[bool] = deque(maxlen=self._adaptive_window)
        # Runtime fallback flag that workers actually observe.  It
        # starts equal to the user-provided ``opencv_fallback`` and
        # may later flip to False by the adaptive logic.
        self._fallback_active = opencv_fallback

        # Always available as fallback
        self._opencv_detector = OpenCVWeChatDetector()

        # Lazily initialised MNN detector (protected by _init_lock
        # so parallel workers don't race on first-call construction).
        self._mnn_detector: QRDetector | None = None
        self._mnn_init_attempted = False
        self._mnn_init_error: str | None = None
        self._init_lock = threading.Lock()

        # Statistics for diagnostics.  CPython's GIL would make naive
        # ``+= 1`` on dict values effectively atomic today, but
        # that's an implementation detail — we take an explicit lock
        # so the counters stay consistent under future free-threading.
        self._stats_lock = threading.Lock()
        self._stats = {
            "mnn_attempts": 0,
            "mnn_success": 0,
            "mnn_fallbacks": 0,
            "opencv_attempts": 0,
            "opencv_success": 0,
            "opencv_rescues": 0,          # OpenCV decoded a frame MNN missed
            "adaptive_disables": 0,       # times adaptive logic turned fallback off
            "adaptive_enables": 0,        # times it turned back on
        }

    # ── QRDetector interface ──────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectResult:
        """Detect QR code, routing to the appropriate backend."""
        if self._use_mnn:
            mnn = self._get_mnn_detector()
            if mnn is not None:
                with self._stats_lock:
                    self._stats["mnn_attempts"] += 1
                result = mnn.detect(frame)
                if result.text is not None:
                    with self._stats_lock:
                        self._stats["mnn_success"] += 1
                    return result
                # MNN returned no-detect.
                with self._stats_lock:
                    self._stats["mnn_fallbacks"] += 1
                    fallback_active = self._fallback_active
                    # Decide whether this MNN miss should trigger an
                    # OpenCV run.  Three cases:
                    #   1. fallback active (user opt-out not set,
                    #      adaptive not suppressing) → always run.
                    #   2. fallback suppressed but adaptive on:
                    #      run periodically so the controller can
                    #      notice if OpenCV starts saving frames
                    #      again.
                    #   3. user-level fallback=False OR adaptive off
                    #      while suppressed → skip entirely.
                    should_run_opencv = fallback_active
                    is_probe = False
                    if (not fallback_active
                            and self._adaptive_fallback
                            and self._opencv_fallback):
                        self._suppressed_miss_count += 1
                        if (self._suppressed_miss_count
                                >= self._adaptive_probe_interval):
                            should_run_opencv = True
                            is_probe = True
                            self._suppressed_miss_count = 0

                if not should_run_opencv:
                    return result

                if is_probe:
                    logger.debug(
                        "DetectorRouter: probing OpenCV fallback "
                        "(suppressed; %d misses since last probe)",
                        self._adaptive_probe_interval,
                    )
                else:
                    logger.debug(
                        "MNN detector returned no-detect, falling back to OpenCV"
                    )

                # Run OpenCV and record whether it rescued the frame,
                # so the adaptive controller can decide whether the
                # fallback path is still pulling its weight.
                with self._stats_lock:
                    self._stats["opencv_attempts"] += 1
                cv_result = self._opencv_detector.detect(frame)
                rescued = cv_result.text is not None
                if rescued:
                    with self._stats_lock:
                        self._stats["opencv_success"] += 1
                        self._stats["opencv_rescues"] += 1
                self._record_rescue_observation(rescued)
                return cv_result

        # OpenCV-only path (use_mnn=False or MNN unavailable).  No
        # rescue observation here — there is no MNN verdict to rescue.
        with self._stats_lock:
            self._stats["opencv_attempts"] += 1
        result = self._opencv_detector.detect(frame)
        if result.text is not None:
            with self._stats_lock:
                self._stats["opencv_success"] += 1
        return result

    def detect_batch(self, frames: list[np.ndarray]) -> list[DetectResult]:
        """Detect QR codes in a batch of frames.

        Routes each frame through :meth:`detect` so all fallback,
        adaptive, and stats logic is preserved.  True batch inference
        will be added in Milestone 5 when ``MNNQrDetector`` overrides
        ``detect_batch`` with a single multi-frame ``runSession`` call.
        """
        return [self.detect(f) for f in frames]

    # ── Adaptive controller ──────────────────────────────────────

    def _record_rescue_observation(self, rescued: bool) -> None:
        """Feed a (mnn-miss, opencv-rescued?) sample into the controller.

        Runs under ``_stats_lock`` for the window deque, but the
        deque itself is only mutated here, so contention is bounded
        to a single append per frame.
        """
        if not self._adaptive_fallback:
            return

        with self._stats_lock:
            self._rescue_log.append(rescued)
            samples = len(self._rescue_log)
            if samples < self._adaptive_warmup:
                return
            rescue_rate = sum(self._rescue_log) / samples
            active = self._fallback_active

            if active and rescue_rate < self._adaptive_disable_rate:
                self._fallback_active = False
                self._stats["adaptive_disables"] += 1
                logger.info(
                    "DetectorRouter: disabling OpenCV fallback (rescue "
                    "rate %.2f%% over last %d MNN misses)",
                    rescue_rate * 100, samples,
                )
            elif (not active) and rescue_rate >= self._adaptive_enable_rate:
                self._fallback_active = True
                self._stats["adaptive_enables"] += 1
                logger.info(
                    "DetectorRouter: re-enabling OpenCV fallback (rescue "
                    "rate %.2f%% over last %d MNN misses)",
                    rescue_rate * 100, samples,
                )

    def is_available(self) -> bool:
        """At minimum, OpenCV detector should be available."""
        return self._opencv_detector.is_available()

    @property
    def name(self) -> str:
        if self._use_mnn and self._mnn_detector is not None:
            return f"router({self._mnn_detector.name}+opencv)"
        return "router(opencv)"

    # ── MNN lifecycle ─────────────────────────────────────────────

    def _get_mnn_detector(self) -> QRDetector | None:
        """Lazily create and return the MNN detector, or None."""
        if self._mnn_init_attempted:
            return self._mnn_detector

        with self._init_lock:
            # Another thread may have completed init while we waited.
            if self._mnn_init_attempted:
                return self._mnn_detector
            self._mnn_init_attempted = True

            try:
                from .mnn_detector import MNNQrDetector
                det = MNNQrDetector(
                    model_dir=self._mnn_model_dir,
                    backend=self._mnn_backend,
                    use_sr=self._use_sr,
                )
                if det.is_available():
                    self._mnn_detector = det
                    logger.info(
                        "DetectorRouter: MNN detector available (%s)", det.name
                    )
                else:
                    self._mnn_init_error = (
                        "MNN detector not available (models missing or MNN "
                        "not installed). Install: pip install 'qrstream[mnn]'"
                    )
                    logger.warning(
                        "DetectorRouter: %s — falling back to OpenCV",
                        self._mnn_init_error,
                    )
            except ImportError as e:
                self._mnn_init_error = (
                    f"MNN import failed: {e}. "
                    f"Install: pip install 'qrstream[mnn]'"
                )
                logger.warning(
                    "DetectorRouter: %s — falling back to OpenCV",
                    self._mnn_init_error,
                )
            except Exception as e:
                self._mnn_init_error = f"MNN init failed: {e}"
                logger.warning(
                    "DetectorRouter: %s, using OpenCV fallback",
                    self._mnn_init_error,
                )

        return self._mnn_detector

    # ── Diagnostics ───────────────────────────────────────────────

    @property
    def active_detector(self) -> QRDetector:
        """Return the currently active primary detector."""
        if self._use_mnn and self._mnn_detector is not None:
            return self._mnn_detector
        return self._opencv_detector

    @property
    def active_detector_can_crash(self) -> bool:
        """Whether the currently active detector may crash the process."""
        return self.active_detector.DETECTOR_CAN_CRASH

    def get_stats(self) -> dict[str, int]:
        """Return a snapshot of detection statistics for diagnostics."""
        with self._stats_lock:
            return dict(self._stats)

    def get_status_summary(self) -> str:
        """Return a human-readable status string."""
        stats = self.get_stats()
        with self._stats_lock:
            fallback_active = self._fallback_active
            samples = len(self._rescue_log)
            rescue_rate = (
                sum(self._rescue_log) / samples if samples else 0.0
            )
        lines = [f"DetectorRouter: use_mnn={self._use_mnn}"]
        if self._use_mnn:
            if self._mnn_detector:
                lines.append(f"  MNN: available ({self._mnn_detector.name})")
            else:
                lines.append(
                    f"  MNN: unavailable "
                    f"({self._mnn_init_error or 'not initialised'})"
                )
        lines.append(
            f"  OpenCV: "
            f"{'available' if self._opencv_detector.is_available() else 'unavailable'}"
        )
        if self._use_mnn and self._opencv_fallback:
            lines.append(
                f"  Fallback: {'active' if fallback_active else 'suppressed'}"
                f" (adaptive={self._adaptive_fallback}, "
                f"rescue_rate={rescue_rate:.1%} over {samples} samples)"
            )
        lines.append(f"  Stats: {stats}")
        return "\n".join(lines)
