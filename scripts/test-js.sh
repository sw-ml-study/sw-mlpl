#!/usr/bin/env bash
set -euo pipefail
# Node unit tests for the mlpl-web 3D-viz JS helpers (derivation ->
# LaTeX gate, etc.). Not part of the cargo gate; run alongside it to
# catch JS-side regressions like the "Extra close brace" derivation bug.
cd "$(dirname "$0")/../components/web/crates/mlpl-web/js"
node --test
