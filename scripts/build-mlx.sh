#!/usr/bin/env bash
# Build the MLX-aware mlpl-serve + the WASM pages on Apple Silicon. The
# pages are pure WASM (no GPU code); the server gets the mlx feature so
# device("mlx") { } runs on the Apple GPU.
#
# See docs/build-and-workspace-plan.md. For Linux+NVIDIA use build-cuda.sh;
# for a GPU-less host use build-pages.sh (pages only). mlx-rs only builds
# on macOS, so this cannot be cross-built.
#
# Builds go through serial.sh so they never deadlock against another
# cargo/trunk on the shared target/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "build-mlx.sh: Apple Silicon (Darwin/arm64) only (this host is $(uname -s)/$(uname -m))." >&2
    echo "  Linux+NVIDIA -> scripts/build-cuda.sh ; GPU-less -> scripts/build-pages.sh" >&2
    exit 1
fi

echo "==> Building mlpl-serve --features mlx..."
( cd "$ROOT/components/serve" && "$ROOT/scripts/serial.sh" \
    cargo build -p mlpl-serve --features mlx --release )

echo "==> Building WASM pages..."
"$ROOT/scripts/build-pages.sh"

echo "==> Done."
echo "    MLX server: $ROOT/target/release/mlpl-serve"
echo "    Pages:      $ROOT/pages/"
echo "    Run the connect server (bind 0.0.0.0, serve pages/) similarly to scripts/serve-cuda.sh."
