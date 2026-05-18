"""
QR code generation and detection utilities.

- :func:`generate_qr_image` produces a BGR QR image from bytes. It
  supports two payload encodings:

  * base64 (default ``alphanumeric=False``) — output is 7-bit ASCII,
    embedded in QR byte mode.
  * base45 (default ``alphanumeric=True``) — output is the 45-char
    QR alphanumeric alphabet, embedded in QR alphanumeric mode, which
    gives ~29% more payload per frame at the same QR version.

- :func:`try_decode_qr` uses zxing-cpp for robust detection.

QR generation backend — zxing-cpp
    Both generation and detection are handled by ``zxing-cpp`` (C++
    library with Python bindings, ISO 18004 compliant).  Using a single
    native library for both paths eliminates the pure-Python GIL
    bottleneck that existed with the previous ``segno`` backend and
    provides ~3.6× speedup in QR frame rendering (V25: ~6 ms → ~1.7 ms
    per frame).

    The previous backend ``qrcode 8.x`` contained a Galois-Field
    arithmetic bug (``glog(0)`` crash); ``segno`` fixed this but
    remained GIL-bound.  ``zxing-cpp`` has neither issue.

History note — OpenCV QRCodeEncoder is not used
    OpenCV 4.13's Python-binding QRCodeEncoder has byte-mode capacity
    ~68% of the ISO table, so any auto-sized payload triggered a
    silent fallback to the ``qrcode`` library.  Micro-benchmarks showed
    the OpenCV path was not actually faster either (both ~40 ms/frame
    at V20).  Removing the OpenCV path eliminates the silent fallback
    and keeps the requested QR version stable.

History note — why we no longer emit COBS payloads
    The pre-0.6 "binary_qr" mode passed ``cobs(data).decode('latin-1')``
    as a Python string to ``qrcode.add_data``. The `qrcode` library
    internally UTF-8-encodes strings, which doubles every byte >= 0x80,
    overflowing the requested QR version and silently upgrading it
    (e.g. V20 -> V25). base45 avoids this by producing pure ASCII and
    using QR alphanumeric mode directly.  COBS support (both encode and
    decode) was removed in v0.10.

History note — WeChatQRCode replaced by zxing-cpp (v0.9)
    ``cv2.wechat_qrcode_WeChatQRCode`` had two fatal problems:
    (1) native SIGSEGV/SIGTRAP on noisy camera frames
        (opencv_contrib#3570, unfixed), requiring a subprocess sandbox.
    (2) extreme latency outliers (mean 4–10× higher than median) due to
        internal retry paths in its bundled zxing code.
    Benchmarks on real 4K phone captures showed zxing-cpp achieves
    equivalent detection rate (≤0.1% difference) at 4–10× the speed
    with negligible per-frame variance and no crash risk.
"""

import base64 as _b64lib

import cv2
import numpy as np
import zxingcpp

# zxing-cpp is reentrant and does not crash on noisy inputs.
# No subprocess sandbox is needed.
DETECTOR_CAN_CRASH: bool = False

# Map ec_level int (0=L,1=M,2=Q,3=H) to zxing-cpp error-correction string.
_EC_MAP: dict[int, str] = {0: 'L', 1: 'M', 2: 'Q', 3: 'H'}


# ── QR Generation ────────────────────────────────────────────────

def _encode_qr_payload(data: bytes,
                       alphanumeric: bool | None = None) -> tuple[str, bool]:
    """Encode packed protocol bytes into the ASCII QR payload string."""
    if alphanumeric is None:
        use_alphanumeric = True
    else:
        use_alphanumeric = bool(alphanumeric)

    if use_alphanumeric:
        # Import lazily so tests that stub protocol still work.
        from .protocol import base45_encode
        payload = base45_encode(data).decode("ascii")
    else:
        payload = _b64lib.b64encode(data).decode("ascii")
    return payload, use_alphanumeric


def generate_qr_image(data: bytes, ec_level: int = 1,
                      box_size: int = 10, border: float = 4,
                      version: int | None = None,
                      alphanumeric: bool | None = None,
                      auto_mask: bool = False) -> np.ndarray:
    """Generate a QR code image from binary data.

    Args:
        data: Raw bytes to encode (a packed protocol block).
        ec_level: Error correction level (0=L, 1=M, 2=Q, 3=H).
        box_size: Pixel size of each QR module.
        border: Quiet-zone border width in QR modules.
        version: QR code version 1-40. If the encoded payload does not
            fit at the requested version, zxing-cpp raises
            ``ValueError``.  Pass ``None`` to let the library choose
            the smallest version that fits.
        alphanumeric: When True (default), encode via base45 into QR
            alphanumeric mode (higher density). When False, encode via
            base64 into QR byte mode.
        auto_mask: Accepted for API compatibility; ignored.  zxing-cpp
            always evaluates all 8 mask patterns in native C++ and
            picks the optimal one.  The cost is negligible (~0.9 ms
            total at V25) so there is no performance reason to skip it.

    Returns:
        BGR numpy array suitable for OpenCV.
    """
    payload, use_alphanumeric = _encode_qr_payload(
        data, alphanumeric=alphanumeric)
    return _render_qr(payload, ec_level, box_size, border, version,
                      use_alphanumeric, auto_mask)


def generate_qr_module_image(data: bytes, ec_level: int = 1,
                             border: float = 4,
                             version: int | None = None,
                             alphanumeric: bool | None = None,
                             auto_mask: bool = False) -> np.ndarray:
    """Generate a 1-pixel-per-module grayscale QR image.

    The returned image includes the quiet zone and uses pure ``0``/``255``
    values. It is intended for display-mode caching; callers can upscale it
    with nearest-neighbour interpolation at playback time.
    """
    payload, use_alphanumeric = _encode_qr_payload(
        data, alphanumeric=alphanumeric)
    return _render_qr_gray(payload, ec_level, 1, border, version,
                           use_alphanumeric, auto_mask)


def _render_qr(payload: str, ec_level: int, box_size: int,
               border: float, version: int | None,
               alphanumeric: bool,
               auto_mask: bool = False) -> np.ndarray:
    """Render an ASCII payload string to a BGR numpy array via zxing-cpp."""
    img = _render_qr_gray(payload, ec_level, box_size, border, version,
                          alphanumeric, auto_mask)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _render_qr_gray(payload: str, ec_level: int, box_size: int,
                    border: float, version: int | None,
                    alphanumeric: bool,
                    auto_mask: bool = False) -> np.ndarray:
    """Render an ASCII payload string to a grayscale numpy array.

    ``payload`` is a plain ASCII string (base45 or base64 encoded).
    zxing-cpp auto-detects the optimal QR encoding mode from the content
    characters (alphanumeric when all chars are in the 45-char QR
    alphanumeric set, byte mode otherwise).

    ``auto_mask`` is accepted for API compatibility but ignored;
    zxing-cpp always evaluates all 8 mask patterns internally (the cost
    is negligible in native C++).
    """
    ec = _EC_MAP.get(ec_level, 'M')
    bs = int(box_size)
    bd = int(border)

    # Build kwargs for create_barcode; omit version to auto-select.
    kwargs: dict = {'ec_level': ec}
    if version is not None:
        kwargs['version'] = version

    bc = zxingcpp.create_barcode(
        payload,
        zxingcpp.BarcodeFormat.QRCode,
        **kwargs,
    )

    # Render to grayscale numpy array at the requested module scale.
    # add_quiet_zones=False: we handle the border manually to support
    # fractional / non-standard quiet-zone widths.
    zimg = bc.to_image(scale=bs, add_quiet_zones=False)
    qr_arr = np.array(zimg, dtype=np.uint8)

    # Add quiet-zone border.
    n = qr_arr.shape[0]
    bd_px = bd * bs
    side = n + 2 * bd_px
    img = np.full((side, side), 255, dtype=np.uint8)
    img[bd_px:bd_px + n, bd_px:bd_px + n] = qr_arr

    return img


# ── QR Detection ─────────────────────────────────────────────────
# Uses zxing-cpp as the primary detector.  zxingcpp.read_barcode is
# reentrant (safe to call from multiple threads simultaneously) and
# does not cache any per-thread state, so no threading.local singleton
# is required.

def try_decode_qr(frame: np.ndarray, qr_detector=None) -> str | None:
    """Decode a QR code from a BGR frame using zxing-cpp.

    ``qr_detector`` is accepted for API compatibility but ignored;
    zxing-cpp is always used.

    Returns the decoded string or None on failure.
    """
    result = try_decode_qr_with_bbox(frame, qr_detector)
    if result is None:
        return None
    return result[0]


def try_decode_qr_with_bbox(
    frame: np.ndarray, qr_detector=None
) -> tuple[str, np.ndarray] | None:
    """Decode a QR code and return both the text and its bounding box.

    ``qr_detector`` is accepted for API compatibility but ignored;
    zxing-cpp is always used.

    Returns ``(decoded_str, bbox)`` on success or ``None`` on no-detect /
    decode failure.  ``bbox`` is a ``(4, 2)`` ``float32`` ndarray of
    QR corner coordinates in clockwise order: TL, TR, BR, BL.

    The bbox lets callers measure the QR's pixel size on the source
    frame, which is the basis for adaptive downscale decisions in
    :mod:`qrstream.decoder`.
    """
    try:
        result = zxingcpp.read_barcode(
            frame,
            formats=zxingcpp.QRCode,
            try_rotate=True,
            # Downscaling is handled by our own adaptive pipeline;
            # disable zxing's internal downscale to avoid double-scaling.
            try_downscale=False,
            try_invert=False,
        )
    except Exception:
        return None

    if result is None or not result.valid or not result.text:
        return None

    text = result.text

    # Build (4, 2) float32 bbox from zxing position corners.
    # zxing-cpp returns corners in the same clockwise TL/TR/BR/BL order
    # as WeChatQRCode, so downstream bbox consumers are unaffected.
    try:
        pos = result.position
        bbox = np.array([
            [pos.top_left.x,     pos.top_left.y],
            [pos.top_right.x,    pos.top_right.y],
            [pos.bottom_right.x, pos.bottom_right.y],
            [pos.bottom_left.x,  pos.bottom_left.y],
        ], dtype=np.float32)
    except Exception:
        # Position unavailable — return a degenerate zero bbox so the
        # caller can still distinguish "decoded" from "no-detect" but
        # will skip module-density estimation.
        bbox = np.zeros((4, 2), dtype=np.float32)

    return (text, bbox)


def reset_strategy_stats():
    """Reset detector state (no-op for zxing-cpp; kept for API compatibility).

    Previously cleared the per-thread WeChatQRCode singleton.
    zxing-cpp is stateless, so nothing needs to be reset.
    """
    pass
