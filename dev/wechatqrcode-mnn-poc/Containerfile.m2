# Milestone 2: CPU formal release & packaging validation.
#
# Validates:
#   * Model files are found via package data path (src/qrstream/detector/models/)
#   * detect_batch default implementation works end-to-end
#   * MNN install via pip install 'qrstream[mnn]' extras
#   * Model path resolution: package > dev > error message
#   * Full fast + slow regression suite passes
#
# Build:
#   podman build -f dev/wechatqrcode-mnn-poc/Containerfile.m2 \
#       -t qrstream-mnn-m2 .
#
# Run:
#   podman run --rm qrstream-mnn-m2
#
# Interactive shell:
#   podman run --rm -it qrstream-mnn-m2 bash

FROM fedora:latest

RUN dnf install -y --setopt=install_weak_deps=False \
        python3.13 python3.13-devel python3-pip \
        gcc gcc-c++ make cmake \
        mesa-libGL libglvnd-glx glib2 libgomp \
        libSM libXext libXrender zlib \
        git \
    && dnf clean all

WORKDIR /app

ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir uv

COPY . .

RUN uv sync --dev --python /usr/bin/python3.13 \
    && uv pip install MNN zxing-cpp

# Verify MNN models are in BOTH locations (package data + dev)
RUN test -f src/qrstream/detector/models/detect.mnn \
    && test -f src/qrstream/detector/models/sr.mnn \
    || (echo "ERROR: Package-data MNN models missing." >&2 && exit 1)
RUN test -f dev/wechatqrcode-mnn-poc/models/mnn/detect.mnn \
    && test -f dev/wechatqrcode-mnn-poc/models/mnn/sr.mnn \
    || (echo "ERROR: Dev MNN models missing." >&2 && exit 1)

CMD ["sh", "-c", "\
    set -e ; \
    echo '=== MNN + zxing-cpp import smoke ===' ; \
    uv run python -c 'import MNN; print(\"MNN version:\", MNN.version())' ; \
    uv run python -c 'import zxingcpp; print(\"zxing-cpp: imported OK\")' ; \
    echo '' ; \
    echo '=== M2: Model path resolution ===' ; \
    uv run python -c '\
from qrstream.detector.mnn_detector import _resolve_model_dir, _PACKAGE_MODEL_DIR, _DEV_MODEL_DIR; \
d = _resolve_model_dir(); \
print(f\"Resolved model dir: {d}\"); \
print(f\"Package dir exists: {_PACKAGE_MODEL_DIR.exists()}\"); \
print(f\"Dev dir exists: {_DEV_MODEL_DIR.exists()}\"); \
assert (d / \"detect.mnn\").exists(), f\"detect.mnn not found in {d}\"' ; \
    echo '' ; \
    echo '=== M2: detect_batch contract ===' ; \
    uv run python -c '\
from qrstream.detector.router import DetectorRouter; \
import numpy as np; \
r = DetectorRouter(use_mnn=True, mnn_backend=\"cpu\"); \
frames = [np.zeros((64,64,3), dtype=np.uint8) for _ in range(4)]; \
results = r.detect_batch(frames); \
assert len(results) == 4, f\"expected 4 results, got {len(results)}\"; \
stats = r.get_stats(); \
assert stats[\"mnn_attempts\"] == 4, f\"expected 4 mnn_attempts, got {stats}\"; \
print(f\"detect_batch OK: {len(results)} results, stats={stats}\")' ; \
    echo '' ; \
    echo '=== M2: detector unit tests ===' ; \
    uv run pytest tests/test_detector.py -v ; \
    echo '' ; \
    echo '=== M2: detector integration (fast) ===' ; \
    uv run pytest tests/test_detector_integration.py -v ; \
    echo '' ; \
    echo '=== M2: cpu_decode contract tests ===' ; \
    uv run pytest tests/test_cpu_decode_contract.py -v ; \
    echo '' ; \
    echo '=== M2: detector integration (slow — MNN CPU end-to-end) ===' ; \
    uv run pytest tests/test_detector_integration.py -v -m slow ; \
    echo '' ; \
    echo '=== M2: full fast suite (regression gate) ===' ; \
    uv run pytest tests/ \
"]
