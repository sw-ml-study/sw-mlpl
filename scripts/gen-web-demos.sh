#!/usr/bin/env bash
# Extract every Demo entry from apps/mlpl-web/src/demos.rs and write
# it as a runnable .mlpl file under work/reg-rs/web-demos/. The reg-rs
# regression suite then covers the web-embedded demos via
# `mlpl-repl -f work/reg-rs/web-demos/<slug>.mlpl`, so any drift
# between the demos.rs inline strings and the on-disk demos/*.mlpl
# files surfaces as a baseline diff instead of a surprise in the
# live REPL.
#
# The awk parser assumes the rustfmt-enforced shape of demos.rs:
#
#     Demo {
#         name: "Name With Spaces",
#         lines: &[
#             "line 1",
#             "line 2 with \"quotes\"",
#         ],
#     },
#
# If that shape changes, update this script in lock-step.
#
# Usage: scripts/gen-web-demos.sh [OUT_DIR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEMOS_RS="$PROJECT_DIR/apps/mlpl-web/src/demos.rs"
OUT_DIR="${1:-$PROJECT_DIR/work/reg-rs/web-demos}"

if [[ ! -f "$DEMOS_RS" ]]; then
    echo "error: $DEMOS_RS not found" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

awk -v outdir="$OUT_DIR" '
BEGIN { mode = "outside"; count = 0 }

mode == "outside" && /^    Demo \{/ {
    mode = "demo"
    name = ""
    next
}

mode == "demo" && /^        name: "/ {
    match($0, /"[^"]*"/)
    name = substr($0, RSTART + 1, RLENGTH - 2)
    next
}

mode == "demo" && /^        lines: &\[/ {
    mode = "lines"
    slug = tolower(name)
    gsub(/ /, "-", slug)
    outfile = outdir "/" slug ".mlpl"
    printf("") > outfile
    next
}

mode == "lines" && /^        \],/ {
    close(outfile)
    count++
    print "wrote " outfile
    mode = "after_lines"
    next
}

mode == "lines" && /^            "/ {
    line = $0
    sub(/^            "/, "", line)
    sub(/",?$/, "", line)
    gsub(/\\"/, "\"", line)
    gsub(/\\\\/, "\\", line)
    printf("%s\n", line) >> outfile
    next
}

mode == "after_lines" && /^    \},/ {
    mode = "outside"
    next
}

END {
    printf("\n%d demos written to %s\n", count, outdir)
}
' "$DEMOS_RS"
