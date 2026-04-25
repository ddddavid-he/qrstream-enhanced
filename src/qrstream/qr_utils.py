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

QR generation backend — segno
    QR images are produced by the ``segno`` library (pure-Python,
    actively maintained, ISO 18004 compliant).  The previous backend
    ``qrcode 8.x`` contained a Galois-Field arithmetic bug
    (``glog(0)`` crash) that triggered when an LT fountain-code block
    happened to produce data whose base45 encoding caused a Reed-Solomon
    block's leading codeword to be 0x00.  ``segno`` has no such bug
    and produces bit-identical QR matrices for well-formed inputs.

    The ``qrcode`` dependency is retained in ``pyproject.toml`` only
    for projects that depend on it transitively; it is no longer
    imported or used by this module.

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
    using QR alphanumeric mode directly. The decoder still recognises
    legacy COBS payloads for backward compatibility.
"""

import base64 as _b64lib
import threading

import cv2
import numpy as np

try:
    import segno
    HAS_SEGNO = True
except ImportError:
    HAS_SEGNO = False

# Future-facing flag. WeChatQRCode (opencv_contrib) has known unfixed
# native crashes in its bundled zxing code (issue opencv_contrib#3570).
# When someone swaps the detector out for a non-crashing implementation
# (e.g. an MNN-based QR pipeline), flip this to False and rely on
# callers to stop spawning sandboxes. Nothing in this module reads the
# flag; it is purely a signal consumed by ``qrstream.decoder`` and
# future code paths that want to know whether crash-isolation is still
# warranted.
DETECTOR_CAN_CRASH: bool = True

# Map ec_level int (0=L,1=M,2=Q,3=H) to segno error-correction letter.
_EC_MAP: dict[int, str] = {0: 'l', 1: 'm', 2: 'q', 3: 'h'}


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
            fit at the requested version, ``segno`` raises
            ``segno.encoder.DataOverflowError``.  Pass ``None`` to let
            the library choose the smallest version that fits.
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

    if not HAS_SEGNO:
        raise RuntimeError(
            "segno library is required for QR generation; "
            "install with `pip install segno`"
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
        payload = base45_encode(data).decode("ascii")
    else:
        payload = _b64lib.b64encode(data).decode("ascii")

    return _render_qr(payload, ec_level, box_size, border, version,
                      use_alphanumeric)


def _render_qr(payload: str, ec_level: int, box_size: int,
               border: float, version: int | None,
               alphanumeric: bool) -> np.ndarray:
    """Render an ASCII payload string to a BGR numpy array via segno.

    ``payload`` is a plain ASCII string (base45 or base64 encoded).
    segno receives it as a str; for the alphanumeric path we explicitly
    request ``mode='alphanumeric'`` so segno never silently falls back
    to byte mode when the string happens to contain only digits.
    """
    ec = _EC_MAP.get(ec_level, 'm')
    mode = 'alphanumeric' if alphanumeric else None

    qr = segno.make(
        payload,
        version=version,
        error=ec,
        mode=mode,
        # boost_error=False: honour the requested EC level exactly,
        # never silently upgrade it (keeps frame size predictable).
        boost_error=False,
        # mask=0: skip segno's per-frame 8-way mask selection loop
        # (ISO 18004 §7.8.3).  segno's mask selector is pure Python
        # and costs ~80% of segno.make()'s wall-clock on V25 frames.
        # ISO 18004 leaves mask selection to the encoder; any mask in
        # 0..7 is standards-compliant.  WeChatQRCode (the decoder we
        # ship) reads whichever mask the generator chose and has no
        # preference among the 8 patterns.  Fixing mask=0 therefore
        # trades an imperceptible scan-quality delta for ~5× single-
        # frame encode speedup on synthetic payloads.
        mask=0,
    )

    # Render via the raw module matrix — faster than encode-to-PNG then
    # re-decode because it skips the PNG compression/decompression cycle.
    #
    # segno.matrix values: 0x00 = light module, 0x01 = dark module,
    # higher values are used for finder/format regions but are still
    # either light (even) or dark (odd) when tested with & 1.
    #
    # Vectorized paint: build a uint8 dark/light mask from the module
    # matrix, expand each module to a bs×bs pixel block via np.repeat,
    # then composite onto a white canvas with the quiet-zone border.
    # This replaces a nested Python for-loop (~10 ms on V25) with a
    # single NumPy broadcast (~0.3 ms).
    mat = qr.matrix          # tuple of bytearrays, one per row
    n = len(mat)             # modules per side (without quiet zone)
    bs = int(box_size)
    bd = int(border)
    side = (n + 2 * bd) * bs

    mat_arr = np.array([list(row) for row in mat], dtype=np.uint8)
    dark = (mat_arr & 1).astype(np.uint8)
    expanded = np.repeat(np.repeat(dark, bs, axis=0), bs, axis=1)

    img = np.full((side, side), 255, dtype=np.uint8)
    inner = slice(bd * bs, (bd + n) * bs)
    img[inner, inner] = np.where(expanded == 1, 0, 255)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# ── QR Detection ─────────────────────────────────────────────────
# Uses WeChatQRCode from opencv-contrib as the primary detector.
# It is faster, more robust, and handles phone-captured screens
# significantly better than OpenCV's built-in QRCodeDetector.

# Per-thread lazy singleton for the WeChatQRCode detector.
# Under ``ThreadPoolExecutor`` each worker thread receives its own
# slot on this ``threading.local()`` object, so the detector is
# initialised once per thread on first call to :func:`try_decode_qr`
# and reused for every subsequent frame that thread processes.  The
# detector is not documented as thread-safe, so we intentionally
# avoid sharing a single instance across threads.
#
# Sentinel semantics are the same as before: ``_UNINIT`` lets us
# distinguish "not yet initialised" from "initialisation failed
# (cv2.error / OSError)", which is stored as ``None``.
_UNINIT = object()
_thread_local = threading.local()


def try_decode_qr(frame: np.ndarray, qr_detector=None) -> str | None:
    """Decode a QR code from a frame using WeChatQRCode.

    Returns the decoded string or None.  Non-UTF-8 payloads (e.g.
    raw-bytes COBS output from some detectors) cause WeChatQR to
    raise UnicodeDecodeError; we swallow that and return None so
    the caller can try alternative detectors if it wants.
    """
    # Lazy-init WeChatQRCode detector (per-thread, for ThreadPoolExecutor)
    detector = getattr(_thread_local, "detector", _UNINIT)
    if detector is _UNINIT:
        try:
            detector = cv2.wechat_qrcode_WeChatQRCode()
        except (cv2.error, OSError):
            detector = None
        _thread_local.detector = detector

    if detector is not None:
        try:
            results, _ = detector.detectAndDecode(frame)
        except UnicodeDecodeError:
            return None
        if results:
            for r in results:
                if r:
                    return r

    return None


def reset_strategy_stats():
    """Reset detector state (useful for testing).

    Rebinds the module-level ``threading.local()`` to a fresh
    instance.  This invalidates the ``detector`` slot on every
    thread at once — mutating a single slot would only clear the
    caller's thread, leaving worker threads with stale cached
    detectors.
    """
    global _thread_local
    _thread_local = threading.local()
