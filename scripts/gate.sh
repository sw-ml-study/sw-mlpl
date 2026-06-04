#!/usr/bin/env bash
set -euo pipefail

# Pre-commit gate, run SERIALLY -- one cargo at a time, via serial.sh's
# global lock (so it can never deadlock against a build or another gate).
#
#   scripts/gate.sh <workspace-dir> <pkg> [pkg...]
#
# <workspace-dir> is the component workspace that owns the packages (this
# repo has no root Cargo.toml; cargo -p must run inside the owning
# workspace, e.g. components/web-demos). For the named packages it runs
# rustfmt --check, clippy -D warnings, and `cargo test`, then sw-checklist.
#
# IMPORTANT: tests must NOT do heavy CPU training (it is slow and wasteful);
# keep interpreter-loop / GPU-fallback training demos out of the default
# test path (see SKIP_DEMOS in the web-demos registry test).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERIAL="$SCRIPT_DIR/serial.sh"

if [ "$#" -lt 2 ]; then
    echo "usage: scripts/gate.sh <workspace-dir> <pkg> [pkg...]" >&2
    exit 2
fi

WS="$1"; shift
PKG_ARGS=()
for p in "$@"; do PKG_ARGS+=(-p "$p"); done

cd "$WS"
echo "=== rustfmt --check (${*}) ==="
"$SERIAL" cargo fmt "${PKG_ARGS[@]}" -- --check

echo "=== clippy -D warnings (${*}) ==="
"$SERIAL" cargo clippy "${PKG_ARGS[@]}" --all-targets --all-features -- -D warnings

echo "=== cargo test (${*}) ==="
"$SERIAL" cargo test "${PKG_ARGS[@]}"

echo "=== sw-checklist ==="
sw-checklist 2>&1 | grep -iE 'Summary' || true

echo "=== gate OK ==="
