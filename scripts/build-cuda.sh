#!/usr/bin/env bash
# Build the CUDA-aware mlpl-serve + the WASM pages on Linux x86_64 with an
# NVIDIA GPU. The pages are pure WASM (no GPU code); the server gets the
# cuda feature so device("cuda") { } runs on the NVIDIA GPU.
#
# See docs/build-and-workspace-plan.md. For Apple use build-mlx.sh; for a
# GPU-less host use build-pages.sh (pages only).
#
# Builds go through serial.sh so they never deadlock against another
# cargo/trunk on the shared target/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-120}"

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "build-cuda.sh: Linux x86_64 only (this host is $(uname -s)/$(uname -m))." >&2
    echo "  Apple Silicon -> scripts/build-mlx.sh ; GPU-less -> scripts/build-pages.sh" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "build-cuda.sh: nvidia-smi not found -- no NVIDIA GPU/driver? The build will" >&2
    echo "  still compile, but device(\"cuda\") needs a real GPU at run time." >&2
fi

echo "==> Building mlpl-serve --features cuda (CUDA_COMPUTE_CAP=$CUDA_COMPUTE_CAP)..."
( cd "$ROOT/components/serve" && "$ROOT/scripts/serial.sh" \
    cargo build -p mlpl-serve --features cuda --release )

echo "==> Building WASM pages..."
"$ROOT/scripts/build-pages.sh"

echo "==> Done."
echo "    CUDA server: $ROOT/target/release/mlpl-serve"
echo "    Pages:       $ROOT/pages/"
echo "    Run the connect server with: scripts/serve-cuda.sh"
