"""
Abstract base for pluggable QR detectors.

Every concrete detector must implement :meth:`detect` and expose
:attr:`DETECTOR_CAN_CRASH` so that the router (and future sandbox /
isolation layers) can decide whether to run the detector in-process
or inside an isolated subprocess.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DetectResult:
    """Result of a single-frame QR detection attempt.

    Attributes:
        text: The decoded QR string, or ``None`` if nothing was found.
        bbox: Optional 4×2 float32 array of corner points (tl, tr, br, bl)
              in the original image coordinate system.  May be ``None``
              when the backend doesn't expose bounding boxes.
        backend: Name of the backend that produced this result
                 (e.g. ``"opencv_wechat"``, ``"mnn_metal"``).
    """
    text: str | None = None
    bbox: np.ndarray | None = None
    backend: str = ""


class QRDetector(abc.ABC):
    """Abstract QR detector interface.

    Subclasses wrap a specific detection backend (OpenCV DNN, MNN, etc.)
    and present a uniform ``detect(frame) -> DetectResult`` contract.

    Thread safety
    -------------
    Implementations must document whether a single instance is safe to
    share across threads.  The current OpenCV WeChatQRCode wrapper uses
    ``threading.local()`` internally — each thread gets its own C++
    detector object.  MNN runners should follow the same pattern or
    create per-thread sessions.
    """

    # Whether this detector is known to be able to crash the process
    # (e.g. native SIGSEGV from OpenCV's bundled ZXing).  When True,
    # the router / sandbox layer may choose to run it in an isolated
    # subprocess.  New detectors should set this to False only after
    # thorough boundary-safety validation.
    DETECTOR_CAN_CRASH: bool = True

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> DetectResult:
        """Attempt QR detection on a single BGR uint8 frame.

        Args:
            frame: BGR uint8 ``np.ndarray`` of shape ``(H, W, 3)``.
                   Implementations must validate shape/dtype and return
                   an empty result on malformed input — never raise.

        Returns:
            A :class:`DetectResult`.  ``text=None`` means no QR found.
        """
        ...

    def detect_batch(self, frames: list[np.ndarray]) -> list[DetectResult]:
        """Attempt QR detection on multiple frames.

        Default implementation loops over :meth:`detect` sequentially.
        Subclasses that support true batch inference (e.g. MNN with
        ``resizeTensor((N,1,H,W))``) may override this for throughput
        gains — see Milestone 5 planning in ``README.md``.

        Args:
            frames: List of BGR uint8 frames, each ``(H, W, 3)``.

        Returns:
            A list of :class:`DetectResult`, one per input frame.
        """
        return [self.detect(f) for f in frames]

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend is loaded and ready to run."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for logging / diagnostics."""
        return self.__class__.__name__
