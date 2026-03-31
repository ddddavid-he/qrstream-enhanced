"""
QR code generation and detection utilities.

- generate_qr_image(): uses OpenCV QRCodeEncoder by default for speed,
  falls back to `qrcode` library for legacy compatibility
- try_decode_qr(): uses WeChatQRCode for robust detection (from opencv-contrib)
"""

import base64

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

# Map ec_level int to qrcode constants
_EC_MAP = {}
if HAS_QRCODE:
    _EC_MAP = {
        0: ERROR_CORRECT_L,
        1: ERROR_CORRECT_M,
        2: ERROR_CORRECT_Q,
        3: ERROR_CORRECT_H,
    }

# Map ec_level int to OpenCV correction level constants
_OPENCV_EC_MAP = {
    0: cv2.QRCODE_ENCODER_CORRECT_LEVEL_L,
    1: cv2.QRCODE_ENCODER_CORRECT_LEVEL_M,
    2: cv2.QRCODE_ENCODER_CORRECT_LEVEL_Q,
    3: cv2.QRCODE_ENCODER_CORRECT_LEVEL_H,
}


# ── QR Generation ────────────────────────────────────────────────

def generate_qr_image(data: bytes, ec_level: int = 1,
                      box_size: int = 10, border: int = 4,
                      version: int | None = None,
                      use_legacy: bool = False,
                      binary_mode: bool = False) -> np.ndarray:
    """Generate a QR code image from binary data.

    By default, data is base64-encoded before embedding in the QR code.
    When binary_mode=True, COBS encoding is used (33% more capacity).

    Returns a BGR numpy array suitable for OpenCV.

    By default uses OpenCV QRCodeEncoder for speed. Set use_legacy=True
    to use the `qrcode` library (slower but offers finer control).

    Args:
        data: Raw bytes to encode
        ec_level: Error correction level (0=L, 1=M, 2=Q, 3=H)
        box_size: Pixel size of each QR module
        border: Module-width of the quiet zone border
        version: QR code version 1-40 (None = auto-fit smallest)
        use_legacy: Force use of qrcode library instead of OpenCV
        binary_mode: Use COBS encoding (skip base64), requires qrcode lib
    """
    if binary_mode:
        return _generate_qr_binary(data, ec_level, box_size, border, version)

    b64 = base64.b64encode(data).decode('ascii')

    if use_legacy and HAS_QRCODE:
        return _generate_qr_legacy(b64, ec_level, box_size, border, version)

    # Default: OpenCV QRCodeEncoder (much faster)
    try:
        return _generate_qr_opencv(b64, ec_level, box_size, border, version)
    except Exception:
        # Fall back to qrcode library if OpenCV encoder fails
        if HAS_QRCODE:
            return _generate_qr_legacy(b64, ec_level, box_size, border, version)
        raise


def _generate_qr_opencv(b64: str, ec_level: int, box_size: int,
                         border: int, version: int | None) -> np.ndarray:
    """Generate QR using OpenCV QRCodeEncoder (fast path)."""
    params = cv2.QRCodeEncoder_Params()
    params.correction_level = _OPENCV_EC_MAP.get(
        ec_level, cv2.QRCODE_ENCODER_CORRECT_LEVEL_M)
    if version is not None:
        params.version = version
    params.mode = cv2.QRCODE_ENCODER_MODE_AUTO

    encoder = cv2.QRCodeEncoder.create(params)
    img = encoder.encode(b64)
    if img is None:
        raise RuntimeError("OpenCV QR encoder returned None")

    # OpenCV QR images have 1px per module. Scale to match expected size.
    h, w = img.shape[:2]
    scale = box_size

    # Add border by padding
    if border > 0:
        border_px = border * scale
        img_scaled = cv2.resize(img, (w * scale, h * scale),
                                interpolation=cv2.INTER_NEAREST)
        padded = cv2.copyMakeBorder(img_scaled,
                                    border_px, border_px, border_px, border_px,
                                    cv2.BORDER_CONSTANT, value=255)
    else:
        padded = cv2.resize(img, (w * scale, h * scale),
                            interpolation=cv2.INTER_NEAREST)

    # Convert to BGR
    if len(padded.shape) == 2:
        return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    return padded


def _generate_qr_legacy(b64: str, ec_level: int, box_size: int,
                         border: int, version: int | None) -> np.ndarray:
    """Generate QR using qrcode library (legacy path, more control)."""
    qr = qrcode.QRCode(
        version=version,
        error_correction=_EC_MAP.get(ec_level, ERROR_CORRECT_M),
        box_size=box_size,
        border=border,
    )
    qr.add_data(b64)
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color="black", back_color="white")
    img_array = np.array(pil_img.convert('RGB'))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def _generate_qr_binary(data: bytes, ec_level: int, box_size: int,
                          border: int, version: int | None) -> np.ndarray:
    """Generate QR with COBS-encoded binary data. ~33% more capacity than base64.

    Pipeline: raw bytes -> COBS encode (eliminates \\x00) -> latin-1 string -> QR.
    """
    from .protocol import cobs_encode

    cobs_data = cobs_encode(data)
    payload = cobs_data.decode('latin-1')

    # Use OpenCV encoder for speed (COBS data is null-free, safe for OpenCV)
    try:
        params = cv2.QRCodeEncoder_Params()
        params.correction_level = _OPENCV_EC_MAP.get(
            ec_level, cv2.QRCODE_ENCODER_CORRECT_LEVEL_M)
        if version is not None:
            params.version = version
        params.mode = cv2.QRCODE_ENCODER_MODE_AUTO

        encoder = cv2.QRCodeEncoder.create(params)
        img = encoder.encode(payload)
        if img is None:
            raise RuntimeError("OpenCV QR encoder returned None")

        h, w = img.shape[:2]
        scale = box_size
        if border > 0:
            border_px = border * scale
            img_scaled = cv2.resize(img, (w * scale, h * scale),
                                    interpolation=cv2.INTER_NEAREST)
            padded = cv2.copyMakeBorder(img_scaled,
                                        border_px, border_px, border_px, border_px,
                                        cv2.BORDER_CONSTANT, value=255)
        else:
            padded = cv2.resize(img, (w * scale, h * scale),
                                interpolation=cv2.INTER_NEAREST)

        if len(padded.shape) == 2:
            return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
        return padded
    except Exception:
        if not HAS_QRCODE:
            raise
        qr = qrcode.QRCode(
            version=version,
            error_correction=_EC_MAP.get(ec_level, ERROR_CORRECT_M),
            box_size=box_size,
            border=border,
        )
        qr.add_data(payload)
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

    Returns the decoded string or None.
    """
    # Lazy-init WeChatQRCode detector (per-process, for multiprocessing)
    if not hasattr(try_decode_qr, '_wechat'):
        try:
            try_decode_qr._wechat = cv2.wechat_qrcode_WeChatQRCode()
        except Exception:
            try_decode_qr._wechat = None

    if try_decode_qr._wechat is not None:
        results, _ = try_decode_qr._wechat.detectAndDecode(frame)
        if results:
            for r in results:
                if r:
                    return r

    return None


def reset_strategy_stats():
    """Reset detector state (useful for testing)."""
    if hasattr(try_decode_qr, '_wechat'):
        del try_decode_qr._wechat


def detect_qr_data(frame: np.ndarray, qr_detector=None) -> bytes | None:
    """Detect and decode a QR code from a video frame.

    Returns decoded raw bytes (after base64 decode) or None.
    """
    qr_string = try_decode_qr(frame, qr_detector)
    if qr_string is None:
        return None
    try:
        return base64.b64decode(qr_string)
    except Exception:
        return None
