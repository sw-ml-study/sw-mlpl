#!/usr/bin/env bash
set -euo pipefail
# Publish a literate MLPL Org document to standalone HTML, in batch.
#
#   ./examples/literate/publish.sh [examples/literate/basics.org]
#
# Runs `emacs -Q --batch` (no user init), loads the MLPL Org-babel
# support (elisp/mlpl-all.el -> ob-mlpl, with mlpl-repl auto-resolved),
# evaluates every source block top-to-bottom maintaining :session
# state, bakes the results in, and exports <file>.html beside it.
#
# Requires: emacs, and a `mlpl-repl` binary on PATH or installed at
# ~/.local/softwarewrighter/bin (build with: cargo build -p mlpl-repl).
here="$(cd "$(dirname "$0")" && pwd)"
org_file="${1:-$here/basics.org}"
exec emacs -Q --batch -l "$here/publish.el" "$org_file"
