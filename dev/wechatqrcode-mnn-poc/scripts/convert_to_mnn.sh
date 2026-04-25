#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# Convert WeChatQRCode Caffe models to MNN format.
#
# Prerequisites:
#   - MNNConvert tool (from MNN release or built from source)
#   - Caffe models already downloaded (run fetch_models.sh first)
#
# Output: dev/wechatqrcode-mnn-poc/models/mnn/
#   detect.mnn
#   sr.mnn
#
# Usage (with podman):
#   podman run --rm -v $(pwd):/workspace:Z mnn-builder \
#       bash /workspace/dev/wechatqrcode-mnn-poc/scripts/convert_to_mnn.sh
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CAFFE_DIR="$SCRIPT_DIR/../models/caffe"
MNN_DIR="$SCRIPT_DIR/../models/mnn"
mkdir -p "$MNN_DIR"

# Check prerequisites
if ! command -v mnnconvert &> /dev/null; then
    echo "ERROR: mnnconvert not found in PATH."
    echo "Install MNN tools or add mnnconvert to your PATH."
    echo ""
    echo "Quick install via pip:  pip install MNN"
    echo "Or build from source:  https://github.com/alibaba/MNN"
    exit 1
fi

for f in detect.prototxt detect.caffemodel sr.prototxt sr.caffemodel; do
    if [ ! -f "$CAFFE_DIR/$f" ]; then
        echo "ERROR: Missing $CAFFE_DIR/$f"
        echo "Run fetch_models.sh first."
        exit 1
    fi
done

echo "Converting Caffe models to MNN format..."
echo ""

# ── Detect model ─────────────────────────────────────────────────
echo "1/2: detect.caffemodel -> detect.mnn"
mnnconvert \
    -f CAFFE \
    --modelFile "$CAFFE_DIR/detect.caffemodel" \
    --prototxt "$CAFFE_DIR/detect.prototxt" \
    --MNNModel "$MNN_DIR/detect.mnn" \
    --bizCode qrstream \
    2>&1 | sed 's/^/    /'
echo "    -> $MNN_DIR/detect.mnn"
echo ""

# ── SR model ─────────────────────────────────────────────────────
echo "2/2: sr.caffemodel -> sr.mnn"
mnnconvert \
    -f CAFFE \
    --modelFile "$CAFFE_DIR/sr.caffemodel" \
    --prototxt "$CAFFE_DIR/sr.prototxt" \
    --MNNModel "$MNN_DIR/sr.mnn" \
    --bizCode qrstream \
    2>&1 | sed 's/^/    /'
echo "    -> $MNN_DIR/sr.mnn"
echo ""

echo "Computing SHA256 hashes for MNN models..."
for f in detect.mnn sr.mnn; do
    if [ -f "$MNN_DIR/$f" ]; then
        hash=$(shasum -a 256 "$MNN_DIR/$f" | cut -d' ' -f1)
        echo "  $f: $hash"
    else
        echo "  $f: MISSING (conversion may have failed)"
    fi
done

echo ""
echo "Done. MNN models saved to: $MNN_DIR"
echo ""
echo "Next step: run verify_models.py to compare outputs with OpenCV DNN."
