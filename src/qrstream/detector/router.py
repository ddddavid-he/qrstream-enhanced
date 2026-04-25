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
    ) -> None:
        self._use_mnn = use_mnn
        self._mnn_model_dir = mnn_model_dir
        self._mnn_backend = mnn_backend
        self._use_sr = use_sr

        # Always available as fallback
        self._opencv_detector = OpenCVWeChatDetector()

        # Lazily initialised MNN detector
        self._mnn_detector: QRDetector | None = None
        self._mnn_init_attempted = False
        self._mnn_init_error: str | None = None

        # Statistics for diagnostics
        self._stats = {
            "mnn_attempts": 0,
            "mnn_success": 0,
            "mnn_fallbacks": 0,
            "opencv_attempts": 0,
            "opencv_success": 0,
        }

    # ── QRDetector interface ──────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectResult:
        """Detect QR code, routing to the appropriate backend."""
        if self._use_mnn:
            mnn = self._get_mnn_detector()
            if mnn is not None:
                self._stats["mnn_attempts"] += 1
                result = mnn.detect(frame)
                if result.text is not None:
                    self._stats["mnn_success"] += 1
                    return result
                # MNN returned no-detect — fall through to OpenCV
                self._stats["mnn_fallbacks"] += 1
                logger.debug("MNN detector returned no-detect, falling back to OpenCV")

        # OpenCV fallback (or default path)
        self._stats["opencv_attempts"] += 1
        result = self._opencv_detector.detect(frame)
        if result.text is not None:
            self._stats["opencv_success"] += 1
        return result

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
                logger.info("DetectorRouter: MNN detector available (%s)", det.name)
            else:
                self._mnn_init_error = "MNN detector not available (models missing or MNN not installed)"
                logger.warning("DetectorRouter: %s, using OpenCV fallback", self._mnn_init_error)
        except ImportError as e:
            self._mnn_init_error = f"MNN import failed: {e}"
            logger.warning("DetectorRouter: %s, using OpenCV fallback", self._mnn_init_error)
        except Exception as e:
            self._mnn_init_error = f"MNN init failed: {e}"
            logger.warning("DetectorRouter: %s, using OpenCV fallback", self._mnn_init_error)

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
        """Return detection statistics for diagnostics."""
        return dict(self._stats)

    def get_status_summary(self) -> str:
        """Return a human-readable status string."""
        lines = [f"DetectorRouter: use_mnn={self._use_mnn}"]
        if self._use_mnn:
            if self._mnn_detector:
                lines.append(f"  MNN: available ({self._mnn_detector.name})")
            else:
                lines.append(f"  MNN: unavailable ({self._mnn_init_error or 'not initialised'})")
        lines.append(f"  OpenCV: {'available' if self._opencv_detector.is_available() else 'unavailable'}")
        lines.append(f"  Stats: {self._stats}")
        return "\n".join(lines)
