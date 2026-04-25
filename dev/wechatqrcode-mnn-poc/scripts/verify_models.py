#!/usr/bin/env python3
"""
Verify MNN model outputs against OpenCV DNN (Caffe) baseline.

For each model (detect + sr), this script:
1. Runs inference with OpenCV DNN (Caffe)
2. Runs inference with MNN
3. Compares outputs numerically (max absolute error, cosine similarity)

Usage:
    python verify_models.py --image test_qr.png
    python verify_models.py --image test_qr.png --verbose

Prerequisites:
    - Caffe models in ../models/caffe/
    - MNN models in ../models/mnn/ (run convert_to_mnn.sh first)
    - pip install opencv-contrib-python numpy
    - pip install MNN (for MNN inference)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CAFFE_DIR = SCRIPT_DIR.parent / "models" / "caffe"
MNN_DIR = SCRIPT_DIR.parent / "models" / "mnn"

# SSD detector input size (matching upstream)
SSD_INPUT_W = 300
SSD_INPUT_H = 300


def verify_detector(image_path: str, verbose: bool = False) -> bool:
    """Compare detector outputs between OpenCV DNN and MNN."""
    print("=" * 60)
    print("Detector model verification")
    print("=" * 60)

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read image: {image_path}")
        return False

    img_h, img_w = img.shape[:2]
    print(f"Input image: {img_w}x{img_h}")

    # ── OpenCV DNN (Caffe) ────────────────────────────────────────
    proto_path = str(CAFFE_DIR / "detect.prototxt")
    model_path = str(CAFFE_DIR / "detect.caffemodel")

    if not Path(proto_path).exists() or not Path(model_path).exists():
        print("SKIP: Caffe model files not found (run fetch_models.sh)")
        return False

    print("\n1) OpenCV DNN inference:")
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    resized = cv2.resize(img, (SSD_INPUT_W, SSD_INPUT_H), interpolation=cv2.INTER_CUBIC)
    blob = cv2.dnn.blobFromImage(resized, 1.0 / 255, (SSD_INPUT_W, SSD_INPUT_H),
                                  (0.0, 0.0, 0.0), False, False)
    net.setInput(blob, "data")

    t0 = time.perf_counter()
    opencv_output = net.forward("detection_output")
    t_opencv = (time.perf_counter() - t0) * 1000

    print(f"   Output shape: {opencv_output.shape}")
    print(f"   Inference time: {t_opencv:.1f} ms")

    # Parse detections
    opencv_dets = []
    for row in range(opencv_output.shape[2]):
        det = opencv_output[0, 0, row]
        if det[1] == 1 and det[2] > 1e-5:
            opencv_dets.append({
                "confidence": float(det[2]),
                "bbox": [float(det[3]), float(det[4]), float(det[5]), float(det[6])],
            })
    print(f"   Detections: {len(opencv_dets)}")
    for i, d in enumerate(opencv_dets):
        print(f"     [{i}] conf={d['confidence']:.4f} bbox={d['bbox']}")

    # ── MNN inference ─────────────────────────────────────────────
    mnn_model_path = MNN_DIR / "detect.mnn"
    if not mnn_model_path.exists():
        print("\nSKIP: MNN model not found (run convert_to_mnn.sh)")
        return False

    try:
        import MNN
    except ImportError:
        print("\nSKIP: MNN Python package not installed")
        return False

    print("\n2) MNN inference:")
    interpreter = MNN.Interpreter(str(mnn_model_path))
    config = MNN.ScheduleConfig()
    config.type = MNN.Forward_CPU  # Use CPU for reproducible comparison
    session = interpreter.createSession(config)

    # Preprocess (same as OpenCV DNN)
    input_data = resized.astype(np.float32) / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))  # HWC -> CHW
    input_data = np.expand_dims(input_data, axis=0)     # NCHW

    input_tensor = interpreter.getSessionInput(session)
    tmp_input = MNN.Tensor(
        input_data.shape, MNN.Halide_Type_Float,
        input_data.astype(np.float32),
        MNN.Tensor_DimensionType_Caffe
    )
    input_tensor.copyFrom(tmp_input)

    t0 = time.perf_counter()
    interpreter.runSession(session)
    t_mnn = (time.perf_counter() - t0) * 1000

    output_tensor = interpreter.getSessionOutput(session, "detection_output")
    output_shape = output_tensor.getShape()
    print(f"   Output shape: {output_shape}")
    print(f"   Inference time: {t_mnn:.1f} ms")

    tmp_output = MNN.Tensor(output_shape, MNN.Halide_Type_Float,
                            np.zeros(output_shape, dtype=np.float32),
                            MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)
    mnn_output = np.array(tmp_output.getData(), dtype=np.float32)

    # ── Compare ───────────────────────────────────────────────────
    print("\n3) Comparison:")
    opencv_flat = opencv_output.flatten()
    mnn_flat = mnn_output.flatten()
    min_len = min(len(opencv_flat), len(mnn_flat))

    if min_len == 0:
        print("   WARNING: Empty output from one or both backends")
        return False

    opencv_flat = opencv_flat[:min_len]
    mnn_flat = mnn_flat[:min_len]

    max_abs_err = np.max(np.abs(opencv_flat - mnn_flat))
    mean_abs_err = np.mean(np.abs(opencv_flat - mnn_flat))

    norm_cv = np.linalg.norm(opencv_flat)
    norm_mnn = np.linalg.norm(mnn_flat)
    cosine_sim = np.dot(opencv_flat, mnn_flat) / (norm_cv * norm_mnn + 1e-12)

    print(f"   Max absolute error:  {max_abs_err:.6f}")
    print(f"   Mean absolute error: {mean_abs_err:.6f}")
    print(f"   Cosine similarity:   {cosine_sim:.6f}")
    print(f"   Speedup:             {t_opencv / t_mnn:.2f}x (CPU vs CPU)")

    ok = max_abs_err < 0.01 and cosine_sim > 0.99
    print(f"\n   Result: {'PASS' if ok else 'FAIL'}")
    return ok


def verify_sr(image_path: str, verbose: bool = False) -> bool:
    """Compare SR model outputs between OpenCV DNN and MNN."""
    print("\n" + "=" * 60)
    print("Super-resolution model verification")
    print("=" * 60)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Cannot read image: {image_path}")
        return False

    # Crop a small region to simulate QR ROI
    h, w = img.shape[:2]
    crop_size = min(100, h, w)
    crop = img[:crop_size, :crop_size]
    print(f"Input crop: {crop.shape[1]}x{crop.shape[0]} (grayscale)")

    # ── OpenCV DNN (Caffe) ────────────────────────────────────────
    proto_path = str(CAFFE_DIR / "sr.prototxt")
    model_path = str(CAFFE_DIR / "sr.caffemodel")

    if not Path(proto_path).exists() or not Path(model_path).exists():
        print("SKIP: Caffe SR model files not found")
        return False

    print("\n1) OpenCV DNN inference:")
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    blob = cv2.dnn.blobFromImage(crop, 1.0 / 255, (crop.shape[1], crop.shape[0]),
                                  (0.0,), False, False)
    net.setInput(blob)

    t0 = time.perf_counter()
    opencv_output = net.forward()
    t_opencv = (time.perf_counter() - t0) * 1000

    print(f"   Output shape: {opencv_output.shape}")
    print(f"   Output range: [{opencv_output.min():.4f}, {opencv_output.max():.4f}]")
    print(f"   Inference time: {t_opencv:.1f} ms")

    # ── MNN inference ─────────────────────────────────────────────
    mnn_model_path = MNN_DIR / "sr.mnn"
    if not mnn_model_path.exists():
        print("\nSKIP: MNN SR model not found")
        return False

    try:
        import MNN
    except ImportError:
        print("\nSKIP: MNN not installed")
        return False

    print("\n2) MNN inference:")
    interpreter = MNN.Interpreter(str(mnn_model_path))
    config = MNN.ScheduleConfig()
    config.type = MNN.Forward_CPU
    session = interpreter.createSession(config)

    input_data = crop.astype(np.float32) / 255.0
    input_data = input_data.reshape(1, 1, crop.shape[0], crop.shape[1])

    input_tensor = interpreter.getSessionInput(session)
    interpreter.resizeTensor(input_tensor, list(input_data.shape))
    interpreter.resizeSession(session)

    tmp_input = MNN.Tensor(
        list(input_data.shape), MNN.Halide_Type_Float,
        input_data.astype(np.float32),
        MNN.Tensor_DimensionType_Caffe
    )
    input_tensor.copyFrom(tmp_input)

    t0 = time.perf_counter()
    interpreter.runSession(session)
    t_mnn = (time.perf_counter() - t0) * 1000

    output_tensor = interpreter.getSessionOutput(session)
    output_shape = output_tensor.getShape()
    print(f"   Output shape: {output_shape}")
    print(f"   Inference time: {t_mnn:.1f} ms")

    tmp_output = MNN.Tensor(output_shape, MNN.Halide_Type_Float,
                            np.zeros(output_shape, dtype=np.float32),
                            MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)
    mnn_output = np.array(tmp_output.getData(), dtype=np.float32)
    print(f"   Output range: [{mnn_output.min():.4f}, {mnn_output.max():.4f}]")

    # ── Compare ───────────────────────────────────────────────────
    print("\n3) Comparison:")
    opencv_flat = opencv_output.flatten()
    mnn_flat = mnn_output.flatten()
    min_len = min(len(opencv_flat), len(mnn_flat))

    if min_len == 0:
        print("   WARNING: Empty output")
        return False

    opencv_flat = opencv_flat[:min_len]
    mnn_flat = mnn_flat[:min_len]

    max_abs_err = np.max(np.abs(opencv_flat - mnn_flat))
    mean_abs_err = np.mean(np.abs(opencv_flat - mnn_flat))
    cosine_sim = np.dot(opencv_flat, mnn_flat) / (
        np.linalg.norm(opencv_flat) * np.linalg.norm(mnn_flat) + 1e-12)

    # Check output sizes match
    opencv_h, opencv_w = opencv_output.shape[2], opencv_output.shape[3]
    mnn_h = output_shape[2] if len(output_shape) > 2 else 0
    mnn_w = output_shape[3] if len(output_shape) > 3 else 0
    size_match = (opencv_h == mnn_h and opencv_w == mnn_w)

    print(f"   Size match: {size_match} (OpenCV: {opencv_h}x{opencv_w}, MNN: {mnn_h}x{mnn_w})")
    print(f"   Max absolute error:  {max_abs_err:.6f}")
    print(f"   Mean absolute error: {mean_abs_err:.6f}")
    print(f"   Cosine similarity:   {cosine_sim:.6f}")
    print(f"   Speedup:             {t_opencv / t_mnn:.2f}x (CPU vs CPU)")

    ok = max_abs_err < 0.01 and cosine_sim > 0.99 and size_match
    print(f"\n   Result: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify MNN models against OpenCV DNN baseline")
    parser.add_argument("--image", required=True,
                        help="Test image path (should contain a QR code)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    results = {}
    results["detector"] = verify_detector(args.image, args.verbose)
    results["sr"] = verify_sr(args.image, args.verbose)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL/SKIP"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
