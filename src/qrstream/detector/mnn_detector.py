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

Threading model:
- A single ``MNNQrDetector`` instance is safe to share across threads.
- Each thread lazily creates its own MNN ``Interpreter`` / ``Session``
  via ``threading.local()``.  MNN sessions are not documented as
  thread-safe for concurrent ``runSession`` calls, and the per-thread
  strategy mirrors what ``OpenCVWeChatDetector`` already does for
  ``cv2.wechat_qrcode_WeChatQRCode``.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from .base import QRDetector, DetectResult
from .mnn_backend import MNNBackend, is_mnn_available, select_backend

logger = logging.getLogger(__name__)

# ── Lightweight decoder: zxing-cpp (preferred) ────────────────────
# zxing-cpp is ~3× faster than WeChatQRCode on MNN-cropped regions
# (survey: P50 ~10 ms vs ~18 ms) and hits ~97% of WeChatQR's decode
# rate on real phone recordings.  The few frames it misses are caught
# by the WeChatQRCode fallback inside _cpu_decode, so end-to-end
# accuracy is never sacrificed for speed.
try:
    import zxingcpp as _zxingcpp
    _HAS_ZXING_CPP = True
except ImportError:
    _zxingcpp = None  # type: ignore[assignment]
    _HAS_ZXING_CPP = False

# Default model paths.
#
# These .mnn files are produced by MNNConvert from the original
# Caffe models — see ``dev/wechatqrcode-mnn-poc/Containerfile.m0``
# for the conversion recipe.
#
# Search order for model files:
#   1. Explicit ``model_dir`` argument to MNNQrDetector
#   2. ``QRSTREAM_MNN_MODEL_DIR`` environment variable
#   3. Package data:  ``src/qrstream/detector/models/``
#      (bundled inside the installed wheel)
#   4. Development tree: ``dev/wechatqrcode-mnn-poc/models/mnn/``
#      (for local development before ``pip install``)
_PACKAGE_MODEL_DIR = Path(__file__).resolve().parent / "models"
_DEV_MODEL_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "dev" / "wechatqrcode-mnn-poc" / "models" / "mnn"
)


def _resolve_model_dir() -> Path:
    """Resolve the model directory using the search order above."""
    import os

    env_dir = os.environ.get("QRSTREAM_MNN_MODEL_DIR")
    if env_dir:
        p = Path(env_dir)
        if (p / _DETECT_MODEL_NAME).exists():
            return p

    if (_PACKAGE_MODEL_DIR / _DETECT_MODEL_NAME).exists():
        return _PACKAGE_MODEL_DIR

    if (_DEV_MODEL_DIR / _DETECT_MODEL_NAME).exists():
        return _DEV_MODEL_DIR

    # Return the package directory as default even if model is missing;
    # the caller will get a clear "model not found" error later.
    return _PACKAGE_MODEL_DIR

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

# Quiet-zone padding ratio applied to every detected bbox before
# handing the crop to the CPU ZXing decoder.
#
# ISO/IEC 18004 requires a 4-module quiet zone around the code.
# MNN's SSD detector returns a **tight** bbox that hugs the outer
# finder modules; cropping on that bbox strips the quiet zone and
# ZXing's Finder Pattern scanner inside the crop fails on every
# frame (see dev/wechatqrcode-mnn-poc/results/detect_breakdown_report.md:
# `pad=0%` → 0/100, `pad=5%` → 93/100, `pad>=15%` → 95/100 — the
# 95/100 figure matches OpenCV full-frame decode, i.e. we hit the
# upper bound).  15% of the bbox's short edge is comfortably more
# than 4 modules for every QR version we generate (V25 => 117×117
# modules; 4 modules is ~3.4% of the code's width) and leaves
# headroom for bbox jitter.
_QUIET_ZONE_PAD_RATIO = 0.15

# Sentinel used by per-thread lazy-init guards to distinguish
# "not yet attempted" from "attempted but failed (cached None)".
_UNINIT = object()


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
        self._model_dir = Path(model_dir) if model_dir else _resolve_model_dir()
        self._use_sr = use_sr
        self._init_lock = threading.Lock()
        self._initialized = False
        self._init_error: str | None = None

        # Per-thread MNN runtime objects.  Each thread lazily builds
        # its own (interpreter, session) pair on first detect() call.
        self._tls = threading.local()

        # Resolved at first _ensure_init() call
        self._backend: MNNBackend | None = None
        self._has_sr_model: bool = False
        self._detect_model_path: str | None = None
        self._sr_model_path: str | None = None

        # Resolve backend
        if isinstance(backend, str):
            self._requested_backend = MNNBackend.from_string(backend)
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

        detect_sess = self._get_thread_detect_session()
        if detect_sess is None:
            return DetectResult(backend=self.name)

        try:
            bboxes = self._run_detector(frame, detect_sess)
        except Exception:
            logger.debug("MNNQrDetector: detector inference failed", exc_info=True)
            return DetectResult(backend=self.name)

        if not bboxes:
            return DetectResult(backend=self.name)

        # Process each detected QR region
        img_h, img_w = frame.shape[:2]
        sr_sess = self._get_thread_sr_session() if self._use_sr else None

        for bbox in bboxes:
            # Safety rule #2: clamp bbox to image bounds
            roi = _clamp_bbox(bbox, img_w, img_h)
            if roi is None:
                continue

            x0, y0, x1, y1 = roi
            # Tight bbox from the SSD detector hugs the outer finder
            # modules.  Expand by a quiet-zone margin before cropping
            # so ZXing inside ``_cpu_decode`` can actually find the
            # finder pattern (without this, hit rate drops to 0% on
            # real captures — see the constant's docstring for data).
            px0, py0, px1, py1 = _pad_bbox(
                x0, y0, x1, y1, img_w, img_h, _QUIET_ZONE_PAD_RATIO,
            )
            cropped = frame[py0:py1, px0:px1]
            if cropped.size == 0:
                continue

            # Optionally apply super-resolution on the padded crop.
            # We intentionally use the padded size for the SR gate:
            # SR is aimed at small codes, and quiet-zone padding
            # only inflates the effective size by ~30% on the short
            # edge, so the threshold remains meaningful.
            if sr_sess is not None:
                area = (px1 - px0) * (py1 - py0)
                if area > 0 and int(area ** 0.5) < _SR_MAX_SIZE:
                    try:
                        cropped = self._run_sr(cropped, sr_sess)
                    except Exception:
                        logger.debug(
                            "MNNQrDetector: SR failed, using original crop",
                            exc_info=True,
                        )

            # CPU decode (use OpenCV WeChatQRCode on the cropped region)
            text = self._cpu_decode(cropped)
            if text is None:
                # Safety net: on the off-chance that padding hurt this
                # particular frame (e.g. heavy background clutter), try
                # the original tight crop as a second attempt.  Data
                # suggests this path almost never fires, but the cost
                # is bounded (one extra ZXing call per missed bbox).
                tight = frame[y0:y1, x0:x1]
                if tight.size:
                    text = self._cpu_decode(tight)

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
        """Resolve backend + validate model files once.

        Only the *metadata* (backend choice, model paths) is resolved
        here — the actual ``Interpreter`` / ``Session`` objects are
        built lazily per thread in :meth:`_get_thread_detect_session`
        and :meth:`_get_thread_sr_session`, because MNN sessions are
        not safe for concurrent ``runSession`` calls.
        """
        if self._initialized:
            return self._init_error is None
        with self._init_lock:
            # Double-checked locking: another thread may have finished
            # initialization while we were waiting for the lock.
            if self._initialized:
                return self._init_error is None

            if not is_mnn_available():
                self._init_error = (
                    "MNN Python bindings not installed. "
                    "Install with: pip install 'qrstream[mnn]'  "
                    "(or: pip install MNN)"
                )
                logger.warning("MNNQrDetector: %s", self._init_error)
                self._initialized = True
                return False

            try:
                self._backend = select_backend(
                    self._requested_backend.value if self._requested_backend else None
                )
            except RuntimeError as e:
                self._init_error = str(e)
                logger.warning("MNNQrDetector: %s", self._init_error)
                self._initialized = True
                return False

            # Detector model is required
            detect_path = self._model_dir / _DETECT_MODEL_NAME
            if not detect_path.exists():
                self._init_error = (
                    f"Detector model not found: {detect_path}. "
                    f"Searched: package={_PACKAGE_MODEL_DIR}, "
                    f"dev={_DEV_MODEL_DIR}. "
                    f"You can set QRSTREAM_MNN_MODEL_DIR to override. "
                    f"To convert models, run the M0 container: "
                    f"podman build -f dev/wechatqrcode-mnn-poc/Containerfile.m0 "
                    f"-t qrstream-mnn-m0 ."
                )
                logger.warning("MNNQrDetector: %s", self._init_error)
                self._initialized = True
                return False
            self._detect_model_path = str(detect_path)

            # SR model is optional
            sr_path = self._model_dir / _SR_MODEL_NAME
            if self._use_sr and sr_path.exists():
                self._has_sr_model = True
                self._sr_model_path = str(sr_path)

            logger.info(
                "MNNQrDetector ready: backend=%s, sr_model=%s",
                self._backend.value,
                self._has_sr_model,
            )
            self._initialized = True
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

    # ── Per-thread session accessors ─────────────────────────────

    def _get_thread_detect_session(self):
        """Return this thread's detector (interpreter, session), lazily.

        Returns ``None`` if init is broken.  Caching failures via a
        sentinel prevents hot-looping on broken models.
        """
        det = getattr(self._tls, "detect_session", _UNINIT)
        if det is _UNINIT:
            try:
                det = self._create_session(self._detect_model_path)
            except Exception as e:
                logger.warning(
                    "MNNQrDetector: per-thread detect session creation failed: %s", e
                )
                det = None
            self._tls.detect_session = det
        return det

    def _get_thread_sr_session(self):
        """Return this thread's SR (interpreter, session), lazily."""
        if not self._has_sr_model:
            return None
        sr = getattr(self._tls, "sr_session", _UNINIT)
        if sr is _UNINIT:
            try:
                sr = self._create_session(self._sr_model_path)
            except Exception as e:
                logger.warning(
                    "MNNQrDetector: per-thread SR session creation failed: %s", e
                )
                sr = None
            self._tls.sr_session = sr
        return sr

    # ── Inference methods ─────────────────────────────────────────

    def _run_detector(self, frame: np.ndarray, detect_sess) -> list[np.ndarray]:
        """Run SSD detector and return list of bounding boxes.

        Each bbox is a (4,2) float32 array of corner points.

        Preprocessing matches upstream ``ssd_detector.cpp`` and prototxt:
        - Convert to grayscale (model input is single-channel)
        - Resize using dynamic scale (target area 400×400)
        - Scale to [0, 1] (divide by 255)
        """
        import MNN

        img_h, img_w = frame.shape[:2]
        interpreter, session = detect_sess

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

    def _run_sr(self, crop: np.ndarray, sr_sess) -> np.ndarray:
        """Run super-resolution on a cropped QR region.

        Preprocessing matches upstream ``super_scale.cpp``:
        - Input: grayscale uint8
        - Normalize to [0, 1] (divide by 255)
        - Output: grayscale uint8 at 2× resolution
        """
        import MNN

        interpreter, session = sr_sess

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
        """Decode a QR code from a cropped image region.

        Strategy (M1.75b — WeChatQRCode-free):
          zxing-cpp with multi-binarization retry (4 attempts):
            1. LocalAverage binarizer
            2. GlobalHistogram binarizer
            3. OpenCV adaptive threshold preprocessing
            4. Inverted image

          This replaces the prior zxing→WeChatQR two-tier fallback.
          Multi-binarization closes the 2.7% gap to within 0.2% of
          WeChatQR on real phone recordings (87.3% vs 87.5%), and
          the remaining misses are frames that are unrecoverable by
          any decoder (confirmed by WeChatQR data — its "extra" hits
          are <0.3% of crops, well within LT redundancy margin).

        When zxing-cpp is not installed, returns None for every crop
        (MNN path becomes detect-only; DetectorRouter's opencv_fallback
        at the frame level still provides a safety net).

        API contract: returns ``str | None``.  Failure → ``None``,
        caller retries with tight crop or next bbox.
        """
        if _HAS_ZXING_CPP:
            return self._decode_zxing_cpp(region)
        return None

    @staticmethod
    def _decode_zxing_cpp(region: np.ndarray) -> str | None:
        """Decode using zxing-cpp with multi-binarization retry.

        Mirrors WeChatQRCode's core strategy of trying multiple
        binarization approaches on the same crop.  WeChatQR uses
        4 C++ binarizers (Hybrid → FastWindow → SimpleAdaptive →
        AdaptiveThresholdMean) + inversion retry.  We replicate this
        via zxing-cpp's ``binarizer`` kwarg + OpenCV preprocessing:

          1. LocalAverage (zxing-cpp default — similar to HybridBinarizer)
          2. GlobalHistogram (fast, works for uniform illumination)
          3. Gaussian adaptive threshold via OpenCV (matches WeChatQR's
             AdaptiveThresholdMeanBinarizer which calls cv::adaptiveThreshold)
          4. Inverted image (matches WeChatQR's getInvertedMatrix retry)

        Each attempt costs ~2–3 ms on a typical crop; in the common
        case attempt 1 succeeds and the others never run.
        """
        try:
            if region.ndim == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            elif region.ndim == 2:
                gray = region
            else:
                return None

            if not gray.flags["C_CONTIGUOUS"]:
                gray = np.ascontiguousarray(gray)

            # ── Attempt 1: LocalAverage (default) ─────────────────
            results = _zxingcpp.read_barcodes(gray)
            for r in results:
                if r.text:
                    return r.text

            # ── Attempt 2: GlobalHistogram ─────────────────────────
            try:
                results = _zxingcpp.read_barcodes(
                    gray, binarizer=_zxingcpp.Binarizer.GlobalHistogram,
                )
                for r in results:
                    if r.text:
                        return r.text
            except (AttributeError, TypeError):
                # Older zxing-cpp without Binarizer enum — skip
                pass

            # ── Attempt 3: OpenCV adaptive threshold preprocessing ─
            # Replicates WeChatQR's AdaptiveThresholdMeanBinarizer
            # which calls cv::adaptiveThreshold(ADAPTIVE_THRESH_GAUSSIAN_C).
            h, w = gray.shape[:2]
            if h >= 25 and w >= 25:
                bs = w // 10
                bs = bs + (bs % 2) - 1  # must be odd and >= 3
                if bs >= 3:
                    thresh = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, bs, 10,
                    )
                    results = _zxingcpp.read_barcodes(
                        thresh, binarizer=_zxingcpp.Binarizer.BoolCast,
                    )
                    for r in results:
                        if r.text:
                            return r.text

            # ── Attempt 4: Inverted image ──────────────────────────
            inverted = cv2.bitwise_not(gray)
            results = _zxingcpp.read_barcodes(inverted)
            for r in results:
                if r.text:
                    return r.text

        except Exception:
            logger.debug("MNNQrDetector._decode_zxing_cpp failed", exc_info=True)
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


def _pad_bbox(
    x0: int, y0: int, x1: int, y1: int,
    img_w: int, img_h: int, ratio: float,
) -> tuple[int, int, int, int]:
    """Expand a clamped (x0, y0, x1, y1) bbox by ``ratio`` and re-clamp.

    Padding is computed from the bbox's own short edge — that keeps
    the effective quiet zone proportional to the QR code's on-screen
    size, so both close-up and far-away captures get comparable
    margins.  At least 1 pixel is added on each side whenever the
    requested ratio is > 0, so pathologically tiny bboxes don't
    silently collapse back to pad=0.

    Safety: the result is always re-clamped to image bounds, so this
    can never produce coordinates outside ``[0, img_w]`` / ``[0, img_h]``.
    """
    if ratio <= 0:
        return (x0, y0, x1, y1)

    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return (x0, y0, x1, y1)

    # Pad based on the short edge so tall-narrow / wide-short bboxes
    # still get a roughly square margin.
    short = min(w, h)
    pad = max(1, int(round(short * ratio)))

    nx0 = max(0, x0 - pad)
    ny0 = max(0, y0 - pad)
    nx1 = min(img_w, x1 + pad)
    ny1 = min(img_h, y1 + pad)

    if nx1 <= nx0 or ny1 <= ny0:
        return (x0, y0, x1, y1)
    return (nx0, ny0, nx1, ny1)
