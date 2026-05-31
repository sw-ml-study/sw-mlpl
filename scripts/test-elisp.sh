#!/usr/bin/env bash
set -euo pipefail
# Regression gate for the MLPL Emacs package (elisp/*.el):
#   1. every file has balanced parens (`check-parens`)
#   2. the uber loader mlpl-all.el loads under `emacs -Q` (no init
#      file) and every module is `featurep'.
# Catches the bug class where an unbalanced/never-loaded file
# (mlpl-fold.el, mlpl-org.el) ships broken. Not part of the cargo
# gate; run alongside it.
cd "$(dirname "$0")/.."
exec emacs -Q --batch -l scripts/elisp-check.el
