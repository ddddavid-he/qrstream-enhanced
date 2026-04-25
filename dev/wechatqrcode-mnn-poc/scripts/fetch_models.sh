#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# Fetch WeChatQRCode Caffe model files from opencv_3rdparty releases.
#
# These are the same models that opencv-contrib-python bundles inside
# cv2.wechat_qrcode_WeChatQRCode().
#
# Output: dev/wechatqrcode-mnn-poc/models/caffe/
#   detect.prototxt
#   detect.caffemodel
#   sr.prototxt
#   sr.caffemodel
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/../models/caffe"
mkdir -p "$MODEL_DIR"

# OpenCV 3rd-party download URLs (matching opencv_contrib 4.x)
BASE_URL="https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode"

FILES=(
    "detect.prototxt"
    "detect.caffemodel"
    "sr.prototxt"
    "sr.caffemodel"
)

echo "Downloading WeChatQRCode Caffe models..."
for f in "${FILES[@]}"; do
    dest="$MODEL_DIR/$f"
    if [ -f "$dest" ]; then
        echo "  $f: already exists, skipping"
    else
        echo "  $f: downloading..."
        curl -fSL "$BASE_URL/$f" -o "$dest"
    fi
done

echo ""
echo "Computing SHA256 hashes..."
for f in "${FILES[@]}"; do
    hash=$(shasum -a 256 "$MODEL_DIR/$f" | cut -d' ' -f1)
    echo "  $f: $hash"
done

echo ""
echo "All models downloaded to: $MODEL_DIR"
echo "Next step: run convert_to_mnn.sh to convert to .mnn format."
