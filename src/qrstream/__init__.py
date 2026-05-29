"""QRStream: Encode and decode files via QR code video streams using LT fountain codes."""

from qrstream.decode_session import (
    DecodeSession,
    DecodeSessionResult,
    DecodeSessionSnapshot,
)

try:
    from qrstream._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "DecodeSession",
    "DecodeSessionResult",
    "DecodeSessionSnapshot",
    "__version__",
]
