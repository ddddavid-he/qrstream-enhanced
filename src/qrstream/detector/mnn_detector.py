"""
MNN-based QR detector: SSD detection + optional super-resolution.

Replaces the OpenCV DNN inference path with MNN while keeping the
ZXing / binarizer CPU decode path (via OpenCV WeChatQRCode or a
lightweight fallback decoder).

Safety constraints (from ``fix/wechat-native-crash`` lessons):
- All input frames are shape/dtype validated before touching native code.
- All bbox / ROI coordinates are clamped to image bounds before cropping.
- All tensor shapes are cross-checked against actual array lengths.
- Any anomaly results in ``DetectResult(text=None)`` — never a crash.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from .base import QRDetector, DetectResult
from .mnn_backend import MNNBackend, is_mnn_available, select_backend

logger = logging.getLogger(__name__)

# Default model paths relative to the models directory.
# These will be .mnn files produced by MNNConvert from the original Caffe models.
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "dev" / "wechatqrcode-mnn-poc" / "models"

_DETECT_MODEL_NAME = "detect.mnn"
_SR_MODEL_NAME = "sr.mnn"

# SSD detector: the upstream prototxt defines input as (1, 1, 384, 384)
# grayscale.  However the actual upstream code uses dynamic sizing based
# on target area 400×400.  We keep the model's native 384 as default but
# the code handles dynamic resize via MNN's resizeTensor.
_SSD_DEFAULT_SIZE = 384

# Target detection area — upstream uses 400×400 for scale calculation
_SSD_TARGET_AREA = 400.0 * 400.0

# SR threshold: only apply super-resolution when the QR region
# (sqrt(area)) is smaller than this in pixels.
_SR_MAX_SIZE = 160


class MNNQrDetector(QRDetector):
    """MNN-accelerated QR detector.

    Runs the WeChatQRCode SSD detector and optional super-resolution
    model via MNN, then delegates final QR decode to a CPU path.

    This detector is designed to be process-safe (pure Python + MNN).
    ``DETECTOR_CAN_CRASH`` is set conservatively to True until
    extensive validation confirms boundary safety.
    """

    DETECTOR_CAN_CRASH: bool = True  # conservative default

    def __init__(
        self,
        model_dir: str | Path | None = None,
        backend: MNNBackend | str | None = None,
        use_sr: bool = True,
    ) -> None:
        self._model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._use_sr = use_sr
        self._initialized = False
        self._init_error: str | None = None

        # MNN runtime objects (set during _lazy_init)
        self._detect_session = None
        self._sr_session = None
        self._backend = None

        # Resolve backend
        if isinstance(backend, str):
            try:
                self._requested_backend = MNNBackend(backend.lower())
            except ValueError:
                self._requested_backend = None
        elif isinstance(backend, MNNBackend):
            self._requested_backend = backend
        else:
            self._requested_backend = None

    # ── QRDetector interface ──────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectResult:
        if not self._ensure_init():
            return DetectResult(backend=self.name)

        # Input validation (safety rule #1)
        if not _valid_frame(frame):
            return DetectResult(backend=self.name)

        try:
            bboxes = self._run_detector(frame)
        except Exception:
            logger.debug("MNNQrDetector: detector inference failed", exc_info=True)
            return DetectResult(backend=self.name)

        if not bboxes:
            return DetectResult(backend=self.name)

        # Process each detected QR region
        img_h, img_w = frame.shape[:2]
        for bbox in bboxes:
            # Safety rule #2: clamp bbox to image bounds
            roi = _clamp_bbox(bbox, img_w, img_h)
            if roi is None:
                continue

            x0, y0, x1, y1 = roi
            cropped = frame[y0:y1, x0:x1]
            if cropped.size == 0:
                continue

            # Optionally apply super-resolution
            if self._use_sr and self._sr_session is not None:
                area = (x1 - x0) * (y1 - y0)
                if area > 0 and int(area ** 0.5) < _SR_MAX_SIZE:
                    try:
                        cropped = self._run_sr(cropped)
                    except Exception:
                        logger.debug("MNNQrDetector: SR failed, using original crop", exc_info=True)

            # CPU decode (use OpenCV WeChatQRCode on the cropped region)
            text = self._cpu_decode(cropped)
            if text:
                bbox_pts = np.array([
                    [x0, y0], [x1, y0], [x1, y1], [x0, y1]
                ], dtype=np.float32)
                return DetectResult(text=text, bbox=bbox_pts, backend=self.name)

        return DetectResult(backend=self.name)

    def is_available(self) -> bool:
        if not is_mnn_available():
            return False
        detect_path = self._model_dir / _DETECT_MODEL_NAME
        return detect_path.exists()

    @property
    def name(self) -> str:
        if self._backend:
            return f"mnn_{self._backend.value}"
        return "mnn"

    # ── Lazy initialization ───────────────────────────────────────

    def _ensure_init(self) -> bool:
        """Lazy-init MNN sessions. Returns True if ready."""
        if self._initialized:
            return self._init_error is None
        self._initialized = True

        if not is_mnn_available():
            self._init_error = "MNN not installed"
            logger.warning("MNNQrDetector: %s", self._init_error)
            return False

        try:
            self._backend = select_backend(
                self._requested_backend.value if self._requested_backend else None
            )
        except RuntimeError as e:
            self._init_error = str(e)
            logger.warning("MNNQrDetector: %s", self._init_error)
            return False

        # Load detector model
        detect_path = self._model_dir / _DETECT_MODEL_NAME
        if not detect_path.exists():
            self._init_error = f"Detector model not found: {detect_path}"
            logger.warning("MNNQrDetector: %s", self._init_error)
            return False

        try:
            self._detect_session = self._create_session(str(detect_path))
        except Exception as e:
            self._init_error = f"Failed to load detector model: {e}"
            logger.warning("MNNQrDetector: %s", self._init_error)
            return False

        # Load SR model (optional — detection works without it)
        sr_path = self._model_dir / _SR_MODEL_NAME
        if self._use_sr and sr_path.exists():
            try:
                self._sr_session = self._create_session(str(sr_path))
            except Exception as e:
                logger.warning("MNNQrDetector: SR model load failed (%s), continuing without SR", e)
                self._sr_session = None

        logger.info("MNNQrDetector initialized: backend=%s, sr=%s",
                     self._backend.value, self._sr_session is not None)
        return True

    def _create_session(self, model_path: str):
        """Create an MNN inference session.

        Uses MNN 3.5+ Python API: ``createSession({'backend': '...'})``
        instead of the deprecated ``ScheduleConfig`` class.
        """
        import MNN

        interpreter = MNN.Interpreter(model_path)

        # Map our backend enum to MNN config dict
        # MNN 3.5+ accepts uppercase strings: CPU, METAL, CUDA
        backend_map = {
            MNNBackend.METAL: "METAL",
            MNNBackend.CUDA: "CUDA",
            MNNBackend.OPENCL: "OpenCL",
            MNNBackend.CPU: "CPU",
        }
        backend_name = backend_map.get(self._backend, "CPU")
        session = interpreter.createSession({"backend": backend_name})
        return (interpreter, session)

    # ── Inference methods ─────────────────────────────────────────

    def _run_detector(self, frame: np.ndarray) -> list[np.ndarray]:
        """Run SSD detector and return list of bounding boxes.

        Each bbox is a (4,2) float32 array of corner points.

        Preprocessing matches upstream ``ssd_detector.cpp`` and prototxt:
        - Convert to grayscale (model input is single-channel)
        - Resize using dynamic scale (target area 400×400)
        - Scale to [0, 1] (divide by 255)
        """
        import MNN

        img_h, img_w = frame.shape[:2]
        interpreter, session = self._detect_session

        # Convert to grayscale (model input: 1 channel)
        if frame.ndim == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            return []

        # Dynamic resize — match upstream target area logic
        scale = min(1.0, (_SSD_TARGET_AREA / (img_w * img_h)) ** 0.5)
        det_w = int(img_w * scale)
        det_h = int(img_h * scale)
        resized = cv2.resize(gray, (det_w, det_h), interpolation=cv2.INTER_CUBIC)

        # Normalize to [0, 1], NCHW layout: (1, 1, H, W)
        input_data = resized.astype(np.float32) / 255.0
        input_data = input_data.reshape(1, 1, det_h, det_w)

        # Resize MNN input tensor to match dynamic size
        input_tensor = interpreter.getSessionInput(session)
        interpreter.resizeTensor(input_tensor, (1, 1, det_h, det_w))
        interpreter.resizeSession(session)

        # Feed to MNN
        tmp_tensor = MNN.Tensor(
            (1, 1, det_h, det_w), MNN.Halide_Type_Float,
            input_data, MNN.Tensor_DimensionType_Caffe
        )
        input_tensor.copyFrom(tmp_tensor)
        interpreter.runSession(session)

        # Get output: "detection_output" layer
        output_tensor = interpreter.getSessionOutput(session, "detection_output")
        output_shape = output_tensor.getShape()

        # Safety rule #3: validate tensor shape
        if len(output_shape) < 4:
            logger.debug("MNN detector: unexpected output shape %s", output_shape)
            return []

        num_detections = output_shape[2]
        dim = output_shape[3]  # 6 (MNN) or 7 (OpenCV DNN)

        if num_detections == 0:
            return []

        if dim not in (6, 7):
            logger.debug("MNN detector: expected dim=6 or 7, got %d", dim)
            return []

        # Copy output to numpy
        tmp_output = MNN.Tensor(
            output_shape, MNN.Halide_Type_Float,
            np.zeros(output_shape, dtype=np.float32),
            MNN.Tensor_DimensionType_Caffe
        )
        output_tensor.copyToHostTensor(tmp_output)
        output_data = np.array(tmp_output.getData(), dtype=np.float32)

        # Safety rule #3: cross-check actual data length
        expected_len = num_detections * dim
        if len(output_data) < expected_len:
            logger.debug("MNN detector: data length %d < expected %d",
                         len(output_data), expected_len)
            return []

        output_data = output_data[:expected_len].reshape(num_detections, dim)

        bboxes = []
        for row in range(num_detections):
            det = output_data[row]
            # MNN DetectionOutput outputs dim=6: [class, conf, x0, y0, x1, y1]
            # OpenCV DNN outputs dim=7: [batch, class, conf, x0, y0, x1, y1]
            if dim == 7:
                class_id, confidence = det[1], det[2]
                x0_n, y0_n, x1_n, y1_n = det[3], det[4], det[5], det[6]
            else:  # dim == 6
                class_id, confidence = det[0], det[1]
                x0_n, y0_n, x1_n, y1_n = det[2], det[3], det[4], det[5]

            if class_id != 1 or confidence <= 1e-5:
                continue

            # Convert normalized coords to original image pixel coords
            x0 = x0_n * img_w
            y0 = y0_n * img_h
            x1 = x1_n * img_w
            y1 = y1_n * img_h

            # Return as 4×2 corner array (matching upstream format)
            bbox = np.array([
                [x0, y0], [x1, y0], [x1, y1], [x0, y1]
            ], dtype=np.float32)
            bboxes.append(bbox)

        return bboxes

    def _run_sr(self, crop: np.ndarray) -> np.ndarray:
        """Run super-resolution on a cropped QR region.

        Preprocessing matches upstream ``super_scale.cpp``:
        - Input: grayscale uint8
        - Normalize to [0, 1] (divide by 255)
        - Output: grayscale uint8 at 2× resolution
        """
        import MNN

        interpreter, session = self._sr_session

        # Convert to grayscale if needed
        if crop.ndim == 3 and crop.shape[2] == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        elif crop.ndim == 2:
            gray = crop
        else:
            return crop

        h, w = gray.shape[:2]
        input_data = gray.astype(np.float32) / 255.0
        # NCHW: (1, 1, H, W)
        input_data = input_data.reshape(1, 1, h, w)

        input_tensor = interpreter.getSessionInput(session)
        # Resize input tensor to match actual crop size (tuple required by MNN 3.5)
        interpreter.resizeTensor(input_tensor, (1, 1, h, w))
        interpreter.resizeSession(session)

        tmp_tensor = MNN.Tensor(
            (1, 1, h, w), MNN.Halide_Type_Float,
            input_data, MNN.Tensor_DimensionType_Caffe
        )
        input_tensor.copyFrom(tmp_tensor)
        interpreter.runSession(session)

        output_tensor = interpreter.getSessionOutput(session)
        output_shape = output_tensor.getShape()

        # Expected output: (1, 1, ~2*H, ~2*W)
        if len(output_shape) < 4:
            logger.debug("MNN SR: unexpected output shape %s", output_shape)
            return crop

        out_h, out_w = output_shape[2], output_shape[3]
        tmp_output = MNN.Tensor(
            output_shape, MNN.Halide_Type_Float,
            np.zeros(output_shape, dtype=np.float32),
            MNN.Tensor_DimensionType_Caffe
        )
        output_tensor.copyToHostTensor(tmp_output)
        output_data = np.array(tmp_output.getData(), dtype=np.float32)

        # Convert back to uint8 grayscale
        expected_pixels = out_h * out_w
        if len(output_data) < expected_pixels:
            logger.debug("MNN SR: data length %d < expected %d",
                         len(output_data), expected_pixels)
            return crop

        result = output_data[:expected_pixels].reshape(out_h, out_w)
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        return result

    def _cpu_decode(self, region: np.ndarray) -> str | None:
        """Decode a QR code from a cropped image region using OpenCV.

        Falls back to the standard WeChatQRCode decoder for the
        actual barcode reading — we only replaced the CNN inference
        (detection + SR), not the ZXing decode path.
        """
        try:
            # Ensure the region is in the right format
            if region.ndim == 2:
                # Grayscale — convert to BGR for WeChatQRCode
                region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)

            if not region.flags['C_CONTIGUOUS']:
                region = np.ascontiguousarray(region)

            detector = cv2.wechat_qrcode_WeChatQRCode()
            results, _ = detector.detectAndDecode(region)
            if results:
                for r in results:
                    if r:
                        return r
        except (cv2.error, UnicodeDecodeError, OSError):
            pass
        except Exception:
            logger.debug("MNNQrDetector._cpu_decode failed", exc_info=True)
        return None


# ── Safety helpers ────────────────────────────────────────────────

def _valid_frame(frame: np.ndarray) -> bool:
    """Validate input frame shape, dtype, and memory layout.

    Safety rule #1: reject any input that could cause UB in native code.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return False
    if frame.ndim != 3:
        return False
    if frame.shape[2] not in (3, 4):
        return False
    if frame.dtype != np.uint8:
        return False
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return False
    # Non-contiguous is OK — we'll copy if needed downstream.
    return True


def _clamp_bbox(bbox: np.ndarray, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """Clamp a 4×2 bbox array to valid image bounds.

    Safety rule #2: never trust model output coordinates directly.

    Returns (x0, y0, x1, y1) as integers, or None if the clamped
    region is degenerate (zero area or negative extent).
    """
    if bbox is None or not isinstance(bbox, np.ndarray):
        return None
    if bbox.shape != (4, 2):
        return None

    # Extract axis-aligned bounding rect from corner points
    xs = bbox[:, 0]
    ys = bbox[:, 1]

    x0 = max(0, int(np.floor(xs.min())))
    y0 = max(0, int(np.floor(ys.min())))
    x1 = min(img_w, int(np.ceil(xs.max())))
    y1 = min(img_h, int(np.ceil(ys.max())))

    # Reject degenerate or inverted rects
    if x1 <= x0 or y1 <= y0:
        return None

    return (x0, y0, x1, y1)
