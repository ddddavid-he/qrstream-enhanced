"""
QR code generation and detection utilities.

- generate_qr_image(): uses OpenCV QRCodeEncoder by default for speed,
  falls back to `qrcode` library for legacy compatibility
- try_decode_qr(): multi-strategy QR detection with adaptive early-exit
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
    When binary_mode=True, raw bytes are embedded directly (33% more capacity).

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
        binary_mode: Embed raw bytes directly (skip base64), requires qrcode lib
    """
    if binary_mode:
        # COBS binary mode: works with OpenCV encoder (no qrcode lib required)
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
    target_h = h * scale + 2 * border * scale
    target_w = w * scale + 2 * border * scale

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

    Pipeline: raw bytes → COBS encode (eliminates \\x00) → latin-1 string → QR.
    This avoids OpenCV's null-termination truncation issue while keeping
    overhead to ~0.4% (vs base64's 33%).
    """
    from .protocol import cobs_encode

    cobs_data = cobs_encode(data)
    # Convert to latin-1 string for QR embedding (all bytes 0x01-0xFF, no nulls)
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
        # Fallback to qrcode library
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


# ── QR Detection (Adaptive Strategy) ────────────────────────────

# Strategy statistics for adaptive early-exit.
# After warmup, strategies with low hit rate are skipped.
_WARMUP_THRESHOLD = 50  # frames before activating adaptive mode
_MIN_HIT_RATE = 0.03    # skip strategies below 3% hit rate after warmup

class _StrategyStats:
    """Per-process strategy performance tracker."""
    __slots__ = ('hits', 'attempts', 'total_frames', 'warmup_done')

    def __init__(self):
        self.hits = {'original': 0, 'gray': 0, 'sharp': 0,
                     'thresh': 0, 'upscale': 0}
        self.attempts = {'original': 0, 'gray': 0, 'sharp': 0,
                         'thresh': 0, 'upscale': 0}
        self.total_frames = 0
        self.warmup_done = False

    def record(self, strategy: str, success: bool):
        self.attempts[strategy] = self.attempts.get(strategy, 0) + 1
        if success:
            self.hits[strategy] = self.hits.get(strategy, 0) + 1

    def should_skip(self, strategy: str) -> bool:
        if not self.warmup_done:
            return False
        attempts = self.attempts.get(strategy, 0)
        if attempts == 0:
            return False
        hit_rate = self.hits.get(strategy, 0) / attempts
        return hit_rate < _MIN_HIT_RATE

    def tick(self):
        self.total_frames += 1
        if not self.warmup_done and self.total_frames >= _WARMUP_THRESHOLD:
            self.warmup_done = True


# Module-level stats (reset per-process in multiprocessing workers)
_stats = _StrategyStats()


def reset_strategy_stats():
    """Reset adaptive strategy statistics (useful for testing)."""
    global _stats
    _stats = _StrategyStats()


def preprocess_variants(frame: np.ndarray):
    """Generate preprocessed frame variants to improve QR detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    thresh = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, 10
    )
    return gray, sharpened, thresh


def try_decode_qr(frame: np.ndarray, qr_detector=None) -> str | None:
    """Try multiple preprocessing strategies to decode a QR code from a frame.

    Uses adaptive early-exit: after a warmup period, strategies with very
    low success rates are skipped to save CPU time. This preserves
    reliability for difficult videos while speeding up clean ones.

    Returns the decoded string or None.
    """
    global _stats

    if qr_detector is None:
        qr_detector = cv2.QRCodeDetector()

    _stats.tick()

    # Strategy 1: original frame (always tried)
    _stats.record('original', False)
    data, points, _ = qr_detector.detectAndDecode(frame)
    if data:
        _stats.hits['original'] += 1
        return data

    # Strategy 2-4: preprocessed variants
    strategy_names = ['gray', 'sharp', 'thresh']
    variants = None  # lazy compute

    for i, name in enumerate(strategy_names):
        if _stats.should_skip(name):
            continue
        if variants is None:
            variants = preprocess_variants(frame)
        _stats.record(name, False)
        data, points, _ = qr_detector.detectAndDecode(variants[i])
        if data:
            _stats.hits[name] += 1
            return data

    # Strategy 5: upscaling for small frames
    min_dim = min(frame.shape[:2])
    if min_dim < 800 and not _stats.should_skip('upscale'):
        for scale in [1.5, 2.0]:
            _stats.record('upscale', False)
            resized = cv2.resize(frame, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)
            data, points, _ = qr_detector.detectAndDecode(resized)
            if data:
                _stats.hits['upscale'] += 1
                return data
            gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            data, points, _ = qr_detector.detectAndDecode(gray_resized)
            if data:
                _stats.hits['upscale'] += 1
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
