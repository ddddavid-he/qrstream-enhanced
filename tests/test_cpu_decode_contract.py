"""
M1.75b contract tests for ``MNNQrDetector._cpu_decode``.

Verifies the decode decision tree:
  1. zxing-cpp multi-binarization hits → return text
  2. All binarization attempts miss → return None
  3. zxing-cpp not installed → return None (DetectorRouter handles fallback)

Also verifies edge cases: grayscale input, non-contiguous memory,
empty region, malformed input.

These tests do NOT require MNN — they exercise the CPU decode
path in isolation by calling the static helper methods directly.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest


# ── Helpers ──────────────────────────────────────────────────────

def _make_qr_image(text: str = "HELLO QRSTREAM", size: int = 200) -> np.ndarray:
    """Generate a clean QR code image (BGR, uint8)."""
    try:
        enc = cv2.QRCodeEncoder.create()
        qr = enc.encode(text)
        if qr is not None and qr.size > 0:
            if qr.ndim == 2:
                qr = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
            h, w = qr.shape[:2]
            canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
            y0 = (size - h) // 2
            x0 = (size - w) // 2
            y1 = min(y0 + h, size)
            x1 = min(x0 + w, size)
            canvas[y0:y1, x0:x1] = qr[: y1 - y0, : x1 - x0]
            return canvas
    except Exception:
        pass
    # Fallback: plain white canvas — decoders will return None,
    # which is fine for miss/fallback tests.
    return np.ones((size, size, 3), dtype=np.uint8) * 255


def _zxing_cpp_available() -> bool:
    try:
        import zxingcpp  # noqa: F401
        return True
    except ImportError:
        return False


# ── Import the class under test ──────────────────────────────────

from qrstream.detector.mnn_detector import MNNQrDetector, _HAS_ZXING_CPP


def _make_plain(decode_attempts: int = 1) -> MNNQrDetector:
    """Construct a bare MNNQrDetector for exercising ``_decode_zxing_cpp``.

    ``_decode_zxing_cpp`` only reads ``self._decode_attempts`` off the
    instance; we skip the MNN model init (``_ensure_init``) because the
    decode-only tests never run detection.  Uses ``__new__`` + direct
    attribute assignment to bypass the constructor's path resolution.
    """
    inst = MNNQrDetector.__new__(MNNQrDetector)
    inst._decode_attempts = decode_attempts
    return inst


# ── zxing-cpp multi-binarization tests ───────────────────────────


@pytest.mark.skipif(
    not _zxing_cpp_available(),
    reason="zxing-cpp not installed",
)
class TestDecodeZxingCpp:
    """The zxing-cpp decoder must handle all inputs.

    Since M3 §3.2 the decoder is an instance method (so ``decode_attempts``
    can be configured per-detector).  The tests exercise the default
    single-attempt path unless otherwise specified.
    """

    def _make_detector(self, decode_attempts=1):
        return _make_plain(decode_attempts)

    def test_decodes_clean_qr_bgr(self):
        img = _make_qr_image("ZXING TEST 456")
        result = _make_plain(3)._decode_zxing_cpp(img)
        assert result == "ZXING TEST 456"

    def test_decodes_grayscale_input(self):
        img = _make_qr_image("ZXING GRAY")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = _make_plain(3)._decode_zxing_cpp(gray)
        assert result == "ZXING GRAY"

    def test_returns_none_on_noise(self):
        noise = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert _make_plain(3)._decode_zxing_cpp(noise) is None

    def test_handles_non_contiguous(self):
        img = _make_qr_image("ZXING CONTIG")
        sliced = img[10:190, 10:190]
        assert not sliced.flags["C_CONTIGUOUS"]
        result = _make_plain(3)._decode_zxing_cpp(sliced)
        assert result is None or isinstance(result, str)

    def test_single_attempt_default_still_decodes_clean_qr(self):
        """The M3 single-attempt default must decode clean inputs."""
        img = _make_qr_image("SINGLE ATTEMPT OK")
        result = _make_plain(1)._decode_zxing_cpp(img)
        assert result == "SINGLE ATTEMPT OK"


# ── _cpu_decode decision tree ────────────────────────────────────


class TestCpuDecodeDecisionTree:
    """_cpu_decode: zxing-cpp multi → None."""

    def _make_detector(self) -> MNNQrDetector:
        """Create a detector without initialising MNN sessions."""
        det = MNNQrDetector.__new__(MNNQrDetector)
        # Needed because _cpu_decode / _decode_zxing_cpp read this
        # off self as of M3 §3.2.
        det._decode_attempts = 3  # exercise the full chain in tests
        return det

    @pytest.mark.skipif(
        not _zxing_cpp_available(),
        reason="zxing-cpp not installed",
    )
    def test_zxing_multi_decodes_clean_qr(self):
        """Multi-binarization zxing-cpp must decode a clean QR."""
        det = self._make_detector()
        img = _make_qr_image("MULTI BINARIZE")
        result = det._cpu_decode(img)
        assert result == "MULTI BINARIZE"

    def test_miss_returns_none(self):
        """When all decoders miss, _cpu_decode returns None."""
        det = self._make_detector()
        noise = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = det._cpu_decode(noise)
        assert result is None

    def test_cpu_decode_never_raises(self):
        """_cpu_decode must never raise — always return str | None."""
        det = self._make_detector()
        for region in [
            np.zeros((0, 0, 3), dtype=np.uint8),
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((10, 10), dtype=np.uint8),  # grayscale
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
        ]:
            result = det._cpu_decode(region)
            assert result is None or isinstance(result, str)

    def test_without_zxing_cpp_returns_none(self):
        """When zxing-cpp is not installed, _cpu_decode returns None."""
        det = self._make_detector()
        img = _make_qr_image("NO ZXING")

        import qrstream.detector.mnn_detector as mod
        original = mod._HAS_ZXING_CPP
        try:
            mod._HAS_ZXING_CPP = False
            result = det._cpu_decode(img)
            assert result is None
        finally:
            mod._HAS_ZXING_CPP = original
