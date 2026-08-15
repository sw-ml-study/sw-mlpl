#!/usr/bin/env bash
# Shared helpers for the examples/scripts/* wrappers. Resolves the repo
# layout from this file's own location (so the scripts work from any
# CWD) and lazily builds the release tools the first time they're used.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLES_DIR/.." && pwd)"
BIN_DIR="$EXAMPLES_DIR/bin"

# ensure_tool <crate-and-bin-name> -> prints the path to a release
# binary, building it once if it isn't present. `mlpl-build` (the
# compiler) and `mlpl-repl` (the interpreter) are the two used here;
# both live in the components/cli workspace (this is a multi-workspace
# monorepo with no root Cargo.toml). The root .cargo/config.toml points
# every workspace at the shared repo-root target/, so the built binary
# lands in $REPO_ROOT/target/release regardless.
ensure_tool() {
    local name="$1"
    local path="$REPO_ROOT/target/release/$name"
    if [ ! -x "$path" ]; then
        echo "building $name (first run, release)..." >&2
        ( cd "$REPO_ROOT/components/cli" && cargo build --release -q -p "$name" )
    fi
    printf '%s' "$path"
}

# every_example -> prints each tracked .mlpl under examples/, one per
# line, excluding compiled output under bin/.
every_example() {
    find "$EXAMPLES_DIR" -name '*.mlpl' -not -path "$BIN_DIR/*" | sort
}
