"""
QR code generation and detection utilities.

- generate_qr_code(): uses the `qrcode` library for fine-grained control
- detect_qr_code():  multi-strategy QR detection via OpenCV
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


# ── QR Generation ────────────────────────────────────────────────

def generate_qr_image(data: bytes, ec_level: int = 1,
                      box_size: int = 10, border: int = 4,
                      version: int | None = None) -> np.ndarray:
    """Generate a QR code image from binary data.

    Data is base64-encoded before embedding in the QR code.
    Returns a BGR numpy array suitable for OpenCV.

    Args:
        data: Raw bytes to encode
        ec_level: Error correction level (0=L, 1=M, 2=Q, 3=H)
        box_size: Pixel size of each QR module
        border: Module-width of the quiet zone border
        version: QR code version 1-40 (None = auto-fit smallest)
    """
    b64 = base64.b64encode(data).decode('ascii')

    if HAS_QRCODE:
        qr = qrcode.QRCode(
            version=version,  # None = auto-detect smallest version
            error_correction=_EC_MAP.get(ec_level, ERROR_CORRECT_M),
            box_size=box_size,
            border=border,
        )
        qr.add_data(b64)
        qr.make(fit=True)
        pil_img = qr.make_image(fill_color="black", back_color="white")
        # Convert PIL → numpy BGR
        img_array = np.array(pil_img.convert('RGB'))
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        # Fallback: use OpenCV's QRCodeEncoder
        encoder = cv2.QRCodeEncoder.create()
        img = encoder.encode(b64)
        if img is None:
            raise RuntimeError("OpenCV QR encoder failed")
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# ── QR Detection ─────────────────────────────────────────────────

def preprocess_variants(frame: np.ndarray):
    """Generate preprocessed frame variants to improve QR detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    thresh = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, 10
    )
    return [gray, sharpened, thresh]


def try_decode_qr(frame: np.ndarray, qr_detector=None) -> str | None:
    """Try multiple preprocessing strategies to decode a QR code from a frame.

    Returns the decoded string or None.
    """
    if qr_detector is None:
        qr_detector = cv2.QRCodeDetector()

    # Try original frame first
    data, points, _ = qr_detector.detectAndDecode(frame)
    if data:
        return data

    # Try preprocessed variants
    for variant in preprocess_variants(frame):
        data, points, _ = qr_detector.detectAndDecode(variant)
        if data:
            return data

    # Try scaling up only for small frames (< 800px on shorter side).
    # For high-res frames (e.g. 4K), upscaling is wasteful.
    min_dim = min(frame.shape[:2])
    if min_dim < 800:
        for scale in [1.5, 2.0]:
            resized = cv2.resize(frame, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)
            data, points, _ = qr_detector.detectAndDecode(resized)
            if data:
                return data
            gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            data, points, _ = qr_detector.detectAndDecode(gray_resized)
            if data:
                return data

    return None


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
