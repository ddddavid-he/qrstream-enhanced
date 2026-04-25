"""
Tests for the pluggable QR detector abstraction layer.

Covers:
- QRDetector abstract interface
- OpenCVWeChatDetector basic behaviour
- DetectorRouter fallback logic
- Input validation / boundary safety (from fix/wechat-native-crash constraints)
- MNNQrDetector safety helpers
"""

from __future__ import annotations

import numpy as np
import pytest

from qrstream.detector.base import QRDetector, DetectResult
from qrstream.detector.opencv_wechat import OpenCVWeChatDetector, _valid_frame
from qrstream.detector.router import DetectorRouter
from qrstream.detector.mnn_detector import _valid_frame as _mnn_valid_frame, _clamp_bbox


# ── DetectResult ──────────────────────────────────────────────────

class TestDetectResult:
    def test_empty_result(self):
        r = DetectResult()
        assert r.text is None
        assert r.bbox is None
        assert r.backend == ""

    def test_with_text(self):
        r = DetectResult(text="hello", backend="test")
        assert r.text == "hello"
        assert r.backend == "test"

    def test_frozen(self):
        r = DetectResult(text="a")
        with pytest.raises(AttributeError):
            r.text = "b"  # type: ignore


# ── Input validation (safety rule #1) ────────────────────────────

class TestFrameValidation:
    """Both OpenCV and MNN detectors share frame validation logic."""

    def test_valid_bgr_frame(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        assert _valid_frame(frame)
        assert _mnn_valid_frame(frame)

    def test_valid_bgra_frame(self):
        frame = np.zeros((100, 200, 4), dtype=np.uint8)
        assert _valid_frame(frame)
        assert _mnn_valid_frame(frame)

    def test_none_input(self):
        assert not _valid_frame(None)
        assert not _mnn_valid_frame(None)

    def test_not_ndarray(self):
        assert not _valid_frame("not an array")
        assert not _mnn_valid_frame([1, 2, 3])

    def test_wrong_ndim_2d(self):
        frame = np.zeros((100, 200), dtype=np.uint8)
        assert not _valid_frame(frame)
        assert not _mnn_valid_frame(frame)

    def test_wrong_ndim_4d(self):
        frame = np.zeros((1, 100, 200, 3), dtype=np.uint8)
        assert not _valid_frame(frame)
        assert not _mnn_valid_frame(frame)

    def test_wrong_channels(self):
        frame = np.zeros((100, 200, 2), dtype=np.uint8)
        assert not _valid_frame(frame)
        assert not _mnn_valid_frame(frame)

    def test_wrong_dtype(self):
        frame = np.zeros((100, 200, 3), dtype=np.float32)
        assert not _valid_frame(frame)
        assert not _mnn_valid_frame(frame)

    def test_zero_height(self):
        frame = np.zeros((0, 200, 3), dtype=np.uint8)
        assert not _valid_frame(frame)
        assert not _mnn_valid_frame(frame)

    def test_zero_width(self):
        frame = np.zeros((100, 0, 3), dtype=np.uint8)
        assert not _valid_frame(frame)
        assert not _mnn_valid_frame(frame)


# ── BBox clamping (safety rule #2) ───────────────────────────────

class TestBboxClamping:
    def test_normal_bbox(self):
        bbox = np.array([[10, 20], [90, 20], [90, 80], [10, 80]], dtype=np.float32)
        result = _clamp_bbox(bbox, 100, 100)
        assert result == (10, 20, 90, 80)

    def test_bbox_exceeds_bounds(self):
        bbox = np.array([[-10, -5], [110, -5], [110, 105], [-10, 105]], dtype=np.float32)
        result = _clamp_bbox(bbox, 100, 100)
        assert result == (0, 0, 100, 100)

    def test_degenerate_bbox_zero_area(self):
        bbox = np.array([[50, 50], [50, 50], [50, 50], [50, 50]], dtype=np.float32)
        result = _clamp_bbox(bbox, 100, 100)
        assert result is None

    def test_inverted_bbox(self):
        """Bbox where min > max after clamping."""
        bbox = np.array([[200, 200], [300, 200], [300, 300], [200, 300]], dtype=np.float32)
        # Image is only 100x100, so clamped to edge; x1=min(100, ceil(300))=100, x0=max(0,200)=200
        # x1 <= x0 → None
        result = _clamp_bbox(bbox, 100, 100)
        assert result is None

    def test_none_bbox(self):
        assert _clamp_bbox(None, 100, 100) is None

    def test_wrong_shape(self):
        bbox = np.array([[10, 20], [90, 80]], dtype=np.float32)  # 2x2 not 4x2
        assert _clamp_bbox(bbox, 100, 100) is None

    def test_negative_coords_clamped(self):
        bbox = np.array([[-50, -30], [50, -30], [50, 60], [-50, 60]], dtype=np.float32)
        result = _clamp_bbox(bbox, 100, 100)
        assert result == (0, 0, 50, 60)


# ── OpenCVWeChatDetector ─────────────────────────────────────────

class TestOpenCVWeChatDetector:
    def test_detect_returns_result(self):
        det = OpenCVWeChatDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert isinstance(result, DetectResult)
        # Empty black frame → no QR detected
        assert result.text is None
        assert result.backend == "opencv_wechat"

    def test_detect_invalid_input(self):
        det = OpenCVWeChatDetector()
        # 2D grayscale → should not crash
        frame = np.zeros((100, 100), dtype=np.uint8)
        result = det.detect(frame)
        assert result.text is None

    def test_detect_none_input(self):
        det = OpenCVWeChatDetector()
        result = det.detect(None)
        assert result.text is None

    def test_name(self):
        det = OpenCVWeChatDetector()
        assert det.name == "opencv_wechat"

    def test_can_crash_flag(self):
        det = OpenCVWeChatDetector()
        assert det.DETECTOR_CAN_CRASH is True


# ── DetectorRouter ───────────────────────────────────────────────

class TestDetectorRouter:
    def test_default_uses_opencv(self):
        router = DetectorRouter(use_mnn=False)
        assert "opencv" in router.name

    def test_detect_without_mnn(self):
        router = DetectorRouter(use_mnn=False)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = router.detect(frame)
        assert isinstance(result, DetectResult)
        assert result.text is None  # black frame

    def test_mnn_fallback_when_unavailable(self):
        """When MNN is requested but not available, should fallback to OpenCV."""
        router = DetectorRouter(use_mnn=True, mnn_model_dir="/nonexistent/path")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = router.detect(frame)
        assert isinstance(result, DetectResult)
        # Should still work via OpenCV fallback
        assert result.text is None  # black frame

    def test_stats_tracking(self):
        router = DetectorRouter(use_mnn=False)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        router.detect(frame)
        router.detect(frame)
        stats = router.get_stats()
        assert stats["opencv_attempts"] == 2

    def test_status_summary(self):
        router = DetectorRouter(use_mnn=False)
        summary = router.get_status_summary()
        assert "DetectorRouter" in summary
        assert "use_mnn=False" in summary

    def test_is_available(self):
        router = DetectorRouter(use_mnn=False)
        # Should be available as long as OpenCV works
        assert isinstance(router.is_available(), bool)

    def test_active_detector_can_crash(self):
        router = DetectorRouter(use_mnn=False)
        # OpenCV WeChatQRCode is known to crash
        assert router.active_detector_can_crash is True


# ── Non-contiguous memory (safety rule #1 supplement) ────────────

class TestNonContiguousMemory:
    def test_non_contiguous_frame(self):
        """Non-contiguous frame should still be handled safely."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Slice creates a non-contiguous view
        sliced = frame[::2, ::2, :]
        assert not sliced.flags['C_CONTIGUOUS']

        det = OpenCVWeChatDetector()
        result = det.detect(sliced)
        assert isinstance(result, DetectResult)
        # Should not crash, may or may not detect
