#!/usr/bin/env bash
# Regenerate one runnable .mlpl file per web-playground demo under
# work/reg-rs/web-demos/, from the canonical DEMOS registry (compiled
# by mlpl-web-demos' build.rs from demos.toml). The reg-rs regression
# suite then covers the web-embedded demos via
# `mlpl-repl -f work/reg-rs/web-demos/<slug>.mlpl`, so any drift
# between demos.toml and the on-disk demos surfaces as a baseline diff.
#
# Each generated file opens with a doc header (the demo's name + intro
# prose) so a reader of the extracted file sees what it does; the
# header is emitted by the generator, so it survives regeneration. The
# actual generation lives in the `gen-web-demos` Rust bin (DEMOS is a
# Rust const now, not a hand-parsed source file) -- this script is a
# thin wrapper that points it at the output directory.
#
# Usage: scripts/gen-web-demos.sh [OUT_DIR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${1:-$PROJECT_DIR/work/reg-rs/web-demos}"
MANIFEST="$PROJECT_DIR/components/web-demos/crates/mlpl-web-demos/Cargo.toml"

cargo run --quiet --manifest-path "$MANIFEST" --bin gen-web-demos -- "$OUT_DIR"
