"""
Pluggable QR detector abstraction layer.

Provides a unified interface for different QR detection backends
(OpenCV WeChatQRCode, MNN, etc.) with automatic fallback.

The default detector remains OpenCV WeChatQRCode for backward
compatibility.  The MNN path is opt-in via ``--mnn`` / ``use_mnn=True``.
"""

from .base import QRDetector, DetectResult
from .opencv_wechat import OpenCVWeChatDetector
from .router import DetectorRouter

__all__ = [
    "QRDetector",
    "DetectResult",
    "OpenCVWeChatDetector",
    "DetectorRouter",
]
