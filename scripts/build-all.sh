#!/usr/bin/env bash
set -euo pipefail

# Build BOTH deployable artifacts for a CUDA connect session, SEQUENTIALLY.
#
# Why sequential: every crate in this repo shares one target/ dir (see
# .cargo/config.toml). Two cargo invocations launched at once block on the
# same build lock and deadlock -- producing no output while wall-clock
# burns. This script serialises the two builds so that never happens.
#
#   1. mlpl-serve --features cuda (release) -- the connect-peer binary that
#      runs device("cuda") demos in-process AND serves the web UI.
#   2. pages/ -- the web UI WASM bundle (delegates to build-pages.sh).
#
# Usage:
#   scripts/build-all.sh             # build both
#   scripts/build-all.sh --serve     # serve binary only
#   scripts/build-all.sh --pages     # pages only
#
# CUDA_COMPUTE_CAP defaults to 120 (Blackwell, e.g. RTX 50-series).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-120}"

DO_SERVE=1
DO_PAGES=1
case "${1:-}" in
    --serve) DO_PAGES=0 ;;
    --pages) DO_SERVE=0 ;;
    "") ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
esac

if [ "$DO_SERVE" -eq 1 ]; then
    echo "=== [serve] Building mlpl-serve (--features cuda, release) ==="
    cd "$PROJECT_DIR/components/serve"
    cargo build -p mlpl-serve --features cuda --release
fi

if [ "$DO_PAGES" -eq 1 ]; then
    echo "=== [pages] Building web UI bundle ==="
    "$SCRIPT_DIR/build-pages.sh"
fi

echo "=== build-all done ==="
[ "$DO_SERVE" -eq 1 ] && echo "Serve binary: $PROJECT_DIR/target/release/mlpl-serve"
[ "$DO_PAGES" -eq 1 ] && echo "Pages:        $PROJECT_DIR/pages/"
echo "Restart the connect server, e.g.:"
echo "  pkill -x mlpl-serve; \\"
echo "  $PROJECT_DIR/target/release/mlpl-serve --bind 0.0.0.0:6464 --auth required --static-dir pages"
