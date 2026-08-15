#!/usr/bin/env bash
# Compile MLPL example(s) to NATIVE binaries for THIS host (Mach-O on
# macOS, ELF on Linux -- `mlpl-build` defaults to the host target), into
# the gitignored examples/bin/ directory.
#
# Usage:
#   examples/scripts/compile.sh <file.mlpl>   # one file
#   examples/scripts/compile.sh --all         # every example (best effort)
#
# Not every example lowers -- the compiler covers a subset of the
# language (no repeat/train/for/grad/Model DSL). Files that don't lower
# are reported and skipped; run them with run.sh (the interpreter).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

BUILD="$(ensure_tool mlpl-build)"
mkdir -p "$BIN_DIR"

compile_one() {
    local src="$1"
    local out="$BIN_DIR/$(basename "${src%.mlpl}")"
    if "$BUILD" "$src" -o "$out" 2>/tmp/mlpl-compile-err; then
        printf 'ok    %s -> %s\n' "${src#"$REPO_ROOT"/}" "${out#"$REPO_ROOT"/}"
    else
        printf 'SKIP  %s (%s)\n' "${src#"$REPO_ROOT"/}" "$(head -1 /tmp/mlpl-compile-err)"
        return 1
    fi
}

if [ "${1:-}" = "--all" ]; then
    rc=0
    while read -r f; do compile_one "$f" || rc=0; done < <(every_example)
    exit "$rc"
fi

[ -n "${1:-}" ] || { echo "usage: compile.sh <file.mlpl> | --all" >&2; exit 2; }
compile_one "$1"
