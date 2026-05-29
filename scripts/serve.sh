#!/usr/bin/env bash
set -euo pipefail

# Local dev server for the MLPL web REPL.
#
# Defaults to --release because the dev profile produces a WASM
# bundle that runs MLPL eval ~30x slower than the release one
# (saga 33 step 027). The deployed pages/ build is release, so
# the local server should mirror that behavior for any UX or
# perceived-speed testing. The slower-binary / faster-recompile
# dev profile is still useful when actively iterating on the
# front-end Rust source -- pass `--no-release` (any token other
# than `--release`) to override:
#
#   scripts/serve.sh                # release (default)
#   scripts/serve.sh --no-release   # dev (faster recompile)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEB_DIR="$(dirname "$SCRIPT_DIR")/components/web/crates/mlpl-web"

PORT=9957
PROFILE_FLAG="--release"

if [[ "${1:-}" == "--no-release" ]]; then
    PROFILE_FLAG=""
    shift
fi

cd "$WEB_DIR"
exec trunk serve --port "$PORT" $PROFILE_FLAG "$@"
