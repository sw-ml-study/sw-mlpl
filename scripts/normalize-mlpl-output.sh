#!/usr/bin/env bash
# Preprocess filter for reg-rs baselines of MLPL demos. Feeds on
# stdin, emits normalized output on stdout. Used by each .rgt via
# the `-P`/`--preprocess` flag so the baselines are robust to:
#
#   1. SVG byte-count drift. The mlpl-viz renderer prints
#      `[svg: 76072 bytes -- pass --svg-out <dir> to save]`, and
#      that byte count can shift by a digit if the SVG output
#      format changes in a semantically-equivalent way (e.g. new
#      coordinate precision). Collapse to `[svg: NNNN bytes`.
#
#   2. Float precision beyond 4 decimal places. MLPL's math is
#      deterministic on a single host, but kernel reorderings and
#      cross-host transcendental differences (mac <-> linux) can
#      shift the last few digits of values like
#      `143.87040581201546`. Round to 4 decimals so
#      `143.87040581201546` and `143.8704058120154` compare equal,
#      but a real regression like `143.87 -> 1.02` still surfaces.
#
# Any future normalization that every demo needs goes here. Per-
# demo tweaks should stay out of this shared filter -- add a
# sibling preprocess script or use the .rgt's own `preprocess`
# field to chain multiple filters.
#
# Usage:
#   reg-rs create -t mlpl-foo -c '...' \
#                 -P 'bash scripts/normalize-mlpl-output.sh'

set -euo pipefail

sed -E \
    -e 's/\[svg: [0-9]+ bytes/[svg: NNNN bytes/g' \
    -e 's/([0-9]+\.[0-9]{4})[0-9]+/\1/g'
