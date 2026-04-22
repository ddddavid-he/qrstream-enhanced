"""
QR code generation and detection utilities.

- :func:`generate_qr_image` produces a BGR QR image from bytes. It
  supports two payload encodings:

  * base64 (default ``alphanumeric=False``) — output is 7-bit ASCII,
    embedded in QR byte mode.
  * base45 (default ``alphanumeric=True``) — output is the 45-char
    QR alphanumeric alphabet, embedded in QR alphanumeric mode, which
    gives ~29% more payload per frame at the same QR version.

- :func:`try_decode_qr` uses WeChatQRCode for robust detection.

History note — OpenCV QRCodeEncoder is not used
    OpenCV 4.13's Python-binding QRCodeEncoder has byte-mode capacity
    ~68% of the ISO table, so any auto-sized payload triggered a
    silent fallback to the `qrcode` library. Micro-benchmarks showed
    the OpenCV path was not actually faster either (both ~40 ms/frame
    at V20). Removing the OpenCV path eliminates the silent fallback
    and keeps the requested QR version stable.

History note — why we no longer emit COBS payloads
    The pre-0.6 "binary_qr" mode passed ``cobs(data).decode('latin-1')``
    as a Python string to ``qrcode.add_data``. The `qrcode` library
    internally UTF-8-encodes strings, which doubles every byte >= 0x80,
    overflowing the requested QR version and silently upgrading it
    (e.g. V20 -> V25). base45 avoids this by producing pure ASCII and
    using QR alphanumeric mode directly. The decoder still recognises
    legacy COBS payloads for backward compatibility.
"""

import base64 as _b64lib

import cv2
import numpy as np

try:
    import qrcode
    from qrcode.constants import (
        ERROR_CORRECT_L,
        ERROR_CORRECT_M,
        ERROR_CORRECT_Q,
        ERROR_CORRECT_H,
    )
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False

# Map ec_level int to qrcode library constants.
_EC_MAP: dict[int, int] = {}
if HAS_QRCODE:
    _EC_MAP = {
        0: ERROR_CORRECT_L,
        1: ERROR_CORRECT_M,
        2: ERROR_CORRECT_Q,
        3: ERROR_CORRECT_H,
    }


# ── QR Generation ────────────────────────────────────────────────

def generate_qr_image(data: bytes, ec_level: int = 1,
                      box_size: int = 10, border: float = 4,
                      version: int | None = None,
                      use_legacy: bool = False,
                      binary_mode: bool | None = None,
                      alphanumeric: bool | None = None) -> np.ndarray:
    """Generate a QR code image from binary data.

    Args:
        data: Raw bytes to encode (a packed protocol block).
        ec_level: Error correction level (0=L, 1=M, 2=Q, 3=H).
        box_size: Pixel size of each QR module.
        border: Quiet-zone border width in QR modules.
        version: QR code version 1-40. If the encoded payload does not
            fit at the requested version, the underlying ``qrcode``
            library raises ``qrcode.exceptions.DataOverflowError``.
            Pass ``None`` to let the library choose.
        use_legacy: Accepted for backward compatibility; ignored.
        binary_mode: Deprecated alias for ``alphanumeric``. If both
            are supplied, ``alphanumeric`` wins.
        alphanumeric: When True (default), encode via base45 into QR
            alphanumeric mode (higher density). When False, encode via
            base64 into QR byte mode.

    Returns:
        BGR numpy array suitable for OpenCV.
    """
    del use_legacy  # legacy parameter kept for API stability

    if not HAS_QRCODE:
        raise RuntimeError(
            "qrcode library is required for QR generation; "
            "install with `pip install qrcode[pil]`"
        )

    # Resolve alphanumeric/binary_mode aliases. Default: alphanumeric.
    if alphanumeric is None:
        if binary_mode is None:
            use_alphanumeric = True
        else:
            use_alphanumeric = bool(binary_mode)
    else:
        use_alphanumeric = bool(alphanumeric)

    if use_alphanumeric:
        # Import lazily so tests that stub protocol still work.
        from .protocol import base45_encode
        payload = base45_encode(data)
    else:
        payload = _b64lib.b64encode(data)

    return _render_qr(payload, ec_level, box_size, border, version,
                      use_alphanumeric)


def _render_qr(payload: bytes, ec_level: int, box_size: int,
               border: float, version: int | None,
               alphanumeric: bool) -> np.ndarray:
    """Render an ASCII payload to a BGR numpy array via ``qrcode``.

    ``payload`` must be bytes of pure ASCII (each char < 0x80).  For
    the alphanumeric path we pass it as ``str`` so that the ``qrcode``
    library can pick its alphanumeric encoding mode (mode=2).  For the
    byte-mode path we keep it as ``bytes`` to match the documented
    behaviour of ``qrcode``.
    """
    if alphanumeric:
        qr_input: object = payload.decode("ascii")
    else:
        qr_input = payload

    qr = qrcode.QRCode(
        version=version,
        error_correction=_EC_MAP.get(ec_level, ERROR_CORRECT_M),
        box_size=box_size,
        border=round(border),
    )
    qr.add_data(qr_input)
    # fit=True is required so that ``qrcode`` validates the payload
    # against the selected version; it raises DataOverflowError on
    # overflow rather than producing a corrupt code.
    qr.make(fit=True)

    pil_img = qr.make_image(fill_color="black", back_color="white")
    img_array = np.array(pil_img.convert('RGB'))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


# ── QR Detection ─────────────────────────────────────────────────
# Uses WeChatQRCode from opencv-contrib as the primary detector.
# It is faster, more robust, and handles phone-captured screens
# significantly better than OpenCV's built-in QRCodeDetector.


def try_decode_qr(frame: np.ndarray, qr_detector=None) -> str | None:
    """Decode a QR code from a frame using WeChatQRCode.

    Returns the decoded string or None.  Non-UTF-8 payloads (e.g.
    raw-bytes COBS output from some detectors) cause WeChatQR to
    raise UnicodeDecodeError; we swallow that and return None so
    the caller can try alternative detectors if it wants.
    """
    # Lazy-init WeChatQRCode detector (per-process, for multiprocessing)
    if not hasattr(try_decode_qr, '_wechat'):
        try:
            try_decode_qr._wechat = cv2.wechat_qrcode_WeChatQRCode()
        except (cv2.error, OSError):
            try_decode_qr._wechat = None

    if try_decode_qr._wechat is not None:
        try:
            results, _ = try_decode_qr._wechat.detectAndDecode(frame)
        except UnicodeDecodeError:
            return None
        if results:
            for r in results:
                if r:
                    return r

    return None


def reset_strategy_stats():
    """Reset detector state (useful for testing)."""
    if hasattr(try_decode_qr, '_wechat'):
        del try_decode_qr._wechat
