#!/usr/bin/env bash
# Run an MLPL example, two ways:
#
#   examples/scripts/run.sh <file.mlpl>            # interpret (default)
#   examples/scripts/run.sh --interp <file.mlpl>   # interpret explicitly
#   examples/scripts/run.sh --native <file.mlpl>   # compile to a native
#                                                  # host binary, then run it
#
# Interpreted mode uses `mlpl-repl -f` and supports the whole language.
# Native mode uses `mlpl-build` (host target) into examples/bin/ and
# execs the binary -- fast, self-contained, but only the compiled
# subset (no repeat/train/for/grad/Model DSL).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

mode=interp
case "${1:-}" in
    --native) mode=native; shift ;;
    --interp) mode=interp; shift ;;
esac

src="${1:-}"
[ -n "$src" ] || { echo "usage: run.sh [--interp|--native] <file.mlpl>" >&2; exit 2; }
[ -f "$src" ] || { echo "no such file: $src" >&2; exit 2; }

if [ "$mode" = native ]; then
    BUILD="$(ensure_tool mlpl-build)"
    mkdir -p "$BIN_DIR"
    out="$BIN_DIR/$(basename "${src%.mlpl}")"
    "$BUILD" "$src" -o "$out"
    exec "$out"
else
    REPL="$(ensure_tool mlpl-repl)"
    exec "$REPL" -f "$src"
fi
