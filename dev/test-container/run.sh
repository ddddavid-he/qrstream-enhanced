#!/usr/bin/env bash
# Local test harness for the PyAV backend migration.
#
# Builds dev/test-container/Containerfile (fedora:latest + uv + Tsinghua
# mirror) on each target arch and runs the chosen mode inside it.
#
# Architectures are selected via podman system connections:
#   arm64 -> default local connection (Apple Silicon, applehv machine)
#   amd64 -> "devcloud-amd64" remote connection (configured via
#            `podman system connection add --identity ... devcloud-amd64
#            ssh://root@<host>:<port>/run/podman/podman.sock`)
#
# Modes:
#   fast   (default)  — pytest tests/ -m "not slow and not e2e"
#   e2e               — pytest tests/ -m e2e
#   slow              — pytest tests/ -m slow         (real phone fixtures)
#   all               — fast, e2e, slow sequentially
#
# Arch arg (positional 2):
#   arm64             — run on the default local applehv machine
#   amd64             — run on the devcloud-amd64 remote connection
#   both  (default)   — arm64 first, then amd64
#
# Examples:
#   ./dev/test-container/run.sh                 # fast, both arches
#   ./dev/test-container/run.sh all arm64       # full gate, arm64 only
#   ./dev/test-container/run.sh e2e amd64       # e2e on amd64 only
#
# CI does NOT invoke this script — it is for local verification only.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
IMG="qrstream-test:fedora"
MODE="${1:-fast}"
ARCH="${2:-both}"
REMOTE_CONN="devcloud-amd64"

_run_on() {
    # $1 = arch label for logs
    # $2+ = extra args passed to podman after the global flags
    local arch="$1"; shift
    local prefix=()

    case "${arch}" in
        arm64)
            prefix=(podman)
            ;;
        amd64)
            prefix=(podman --connection "${REMOTE_CONN}")
            ;;
        *)
            echo "[run.sh] internal error: unknown arch '${arch}'" >&2
            exit 2
            ;;
    esac

    echo ""
    echo "[run.sh] =========================================="
    echo "[run.sh]  arch: ${arch}    mode: ${MODE}"
    echo "[run.sh] =========================================="

    echo "[run.sh] ${arch}: building ${IMG} ..."
    "${prefix[@]}" build \
        -t "${IMG}" \
        -f "${REPO_ROOT}/dev/test-container/Containerfile" \
        "${REPO_ROOT}"

    echo "[run.sh] ${arch}: smoke — import av + cv2 + zxing-cpp"
    "${prefix[@]}" run --rm "${IMG}" \
        uv run python -c '
import av, cv2, zxingcpp, numpy as np
print(f"av={av.__version__}  cv2={cv2.__version__}  zxing={zxingcpp.__version__}")
print("smoke OK")
'

    case "${MODE}" in
        fast)
            "${prefix[@]}" run --rm "${IMG}" \
                uv run pytest tests/ -v -m "not slow and not e2e"
            ;;
        e2e)
            "${prefix[@]}" run --rm "${IMG}" \
                uv run pytest tests/ -v -m e2e
            ;;
        slow)
            "${prefix[@]}" run --rm "${IMG}" \
                uv run pytest tests/ -v -m slow
            ;;
        all)
            "${prefix[@]}" run --rm "${IMG}" \
                uv run pytest tests/ -v -m "not slow and not e2e"
            "${prefix[@]}" run --rm "${IMG}" \
                uv run pytest tests/ -v -m e2e
            "${prefix[@]}" run --rm "${IMG}" \
                uv run pytest tests/ -v -m slow
            ;;
        *)
            echo "Usage: $0 [fast|e2e|slow|all] [arm64|amd64|both]" >&2
            exit 1
            ;;
    esac
}

case "${ARCH}" in
    arm64)
        _run_on arm64
        ;;
    amd64)
        _run_on amd64
        ;;
    both)
        _run_on arm64
        _run_on amd64
        ;;
    *)
        echo "Usage: $0 [fast|slow|all|bench] [arm64|amd64|both]" >&2
        exit 1
        ;;
esac
