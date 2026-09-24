#!/usr/bin/env bash
set -euo pipefail
# Publish a literate MLPL Org document to standalone HTML, in batch.
#
#   ./examples/literate/publish.sh [examples/literate/basics.org]
#   ./examples/literate/publish.sh --export-only examples/literate/*.org
#
# Runs `emacs -Q --batch` (no user init), loads the MLPL Org-babel
# support (elisp/mlpl-all.el -> ob-mlpl, with mlpl-repl auto-resolved),
# evaluates every source block top-to-bottom maintaining :session
# state, bakes the results in, and exports <file>.html beside it with
# syntax-colored source blocks (htmlize + mlpl-code.css; see
# publish.el).
#
# --export-only re-exports the committed results without evaluating --
# for re-styling docs whose blocks need hardware this host lacks.
#
# Requires: emacs ($EMACS, then PATH, then the macOS app bundle), a
# `mlpl-repl` binary on PATH or at ~/.local/softwarewrighter/bin (build
# with: cargo build -p mlpl-repl), and for colors htmlize from NonGNU
# ELPA:
#   emacs --batch --eval "(progn (package-refresh-contents) (package-install 'htmlize))"
here="$(cd "$(dirname "$0")" && pwd)"
if [[ "${1:-}" == "--export-only" ]]; then
  export MLPL_PUBLISH_EXPORT_ONLY=1
  shift
fi
emacs_bin="${EMACS:-$(command -v emacs || true)}"
for cand in /Applications/Emacs.app/Contents/MacOS/Emacs "$HOME/Applications/Emacs.app/Contents/MacOS/Emacs"; do
  [[ -n "$emacs_bin" ]] && break
  [[ -x "$cand" ]] && emacs_bin="$cand"
done
[[ -n "$emacs_bin" ]] || { echo "publish.sh: emacs not found (set EMACS)" >&2; exit 1; }
[[ $# -gt 0 ]] || set -- "$here/basics.org"
for org_file in "$@"; do
  "$emacs_bin" -Q --batch -l "$here/publish.el" "$org_file"
done
