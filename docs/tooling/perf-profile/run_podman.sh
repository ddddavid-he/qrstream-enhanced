#!/usr/bin/env bash
# Wrapper that builds the profiling image and runs it, following the
# "use podman for build/test" convention.
#
# Usage:
#   bash docs/tooling/perf-profile/run_podman.sh               # --quick (default)
#   bash docs/tooling/perf-profile/run_podman.sh --sizes 10,100
#   bash docs/tooling/perf-profile/run_podman.sh full          # full suite (1KB–10MB)
set -euo pipefail

cd "$(dirname "$0")/../../.."

IMAGE="qrstream-profile:latest"

echo "==> Ensuring podman machine is running..."
podman machine start 2>/dev/null || true

echo "==> Building $IMAGE ..."
podman build -t "$IMAGE" -f docs/tooling/perf-profile/Containerfile .

mkdir -p docs/tooling/perf-profile/results

# Parse args: 'full' → no flag, everything else → passthrough
EXTRA=("$@")
if [ "${1:-}" = "full" ]; then
    EXTRA=()
elif [ $# -eq 0 ]; then
    EXTRA=("--quick")
fi

echo "==> Running profile suite: ${EXTRA[*]:-<default>}"
podman run --rm \
    -v "$(pwd)/docs/tooling/perf-profile/results:/app/docs/tooling/perf-profile/results:Z" \
    "$IMAGE" \
    python docs/tooling/perf-profile/run_all.py "${EXTRA[@]}"

echo ""
echo "==> Results saved to: $(pwd)/docs/tooling/perf-profile/results/"
ls -la docs/tooling/perf-profile/results/
