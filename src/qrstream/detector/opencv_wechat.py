"""
OpenCV WeChatQRCode detector — wraps the existing ``cv2.wechat_qrcode``
backend behind the :class:`QRDetector` interface.

This is the **default** detector and the stable fallback target for all
other backends.  It must remain fully functional regardless of whether
MNN or any other accelerator is installed.

Thread safety: each thread lazily initialises its own
``WeChatQRCode`` instance via ``threading.local()``, mirroring the
strategy used in ``qr_utils.py`` before the detector abstraction was
introduced.
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from .base import QRDetector, DetectResult

logger = logging.getLogger(__name__)

_UNINIT = object()


class OpenCVWeChatDetector(QRDetector):
    """OpenCV WeChatQRCode detector.

    Wraps ``cv2.wechat_qrcode_WeChatQRCode()`` with per-thread lazy
    initialisation, matching the existing ``qr_utils.py`` semantics.

    This detector is known to crash on certain malformed frames (see
    ``fix/wechat-native-crash``), so ``DETECTOR_CAN_CRASH`` is True.
    """

    DETECTOR_CAN_CRASH: bool = True

    def __init__(self) -> None:
        self._tls = threading.local()
        # Module-level availability flag (set once, read many).
        self._available: bool | None = None

    # ── QRDetector interface ──────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectResult:
        """Detect and decode QR codes from a BGR uint8 frame."""
        # Validate input — never crash on bad data.
        if not _valid_frame(frame):
            return DetectResult(backend=self.name)

        detector = self._get_thread_detector()
        if detector is None:
            return DetectResult(backend=self.name)

        try:
            results, points = detector.detectAndDecode(frame)
        except UnicodeDecodeError:
            return DetectResult(backend=self.name)
        except Exception:
            # Catch-all: cv2.error, segfault-wrapped exceptions, etc.
            logger.debug("OpenCVWeChatDetector: detectAndDecode raised", exc_info=True)
            return DetectResult(backend=self.name)

        if results:
            for idx, r in enumerate(results):
                if r:
                    bbox = None
                    if points is not None and idx < len(points):
                        bbox = np.asarray(points[idx], dtype=np.float32)
                    return DetectResult(text=r, bbox=bbox, backend=self.name)

        return DetectResult(backend=self.name)

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        # Try to construct a detector once to check availability.
        try:
            cv2.wechat_qrcode_WeChatQRCode()
            self._available = True
        except (cv2.error, OSError, AttributeError):
            self._available = False
        return self._available

    @property
    def name(self) -> str:
        return "opencv_wechat"

    # ── Internal helpers ──────────────────────────────────────────

    def _get_thread_detector(self):
        """Return the per-thread WeChatQRCode instance, creating lazily."""
        det = getattr(self._tls, "detector", _UNINIT)
        if det is _UNINIT:
            try:
                det = cv2.wechat_qrcode_WeChatQRCode()
            except (cv2.error, OSError, AttributeError):
                det = None
            self._tls.detector = det
        return det


def _valid_frame(frame: np.ndarray) -> bool:
    """Quick sanity check on the input frame.

    Rejects frames that would cause undefined behaviour in native code.
    """
    if frame is None:
        return False
    if not isinstance(frame, np.ndarray):
        return False
    if frame.ndim != 3:
        return False
    if frame.shape[2] not in (3, 4):
        return False
    if frame.dtype != np.uint8:
        return False
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return False
    return True
