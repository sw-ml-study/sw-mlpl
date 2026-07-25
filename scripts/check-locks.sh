#!/usr/bin/env bash
set -euo pipefail

# Repo-wide Cargo.lock consistency sweep.
#
# Every component directory is its own Cargo workspace with its own
# Cargo.lock, and a lock also pins path-dependencies OWNED BY OTHER
# workspaces. So a manifest change in one component silently strands
# the locks of every downstream workspace until someone builds there
# (the "stale cli lock" incident, 2026-07-25: 14 workspaces drifted).
#
# This script fails fast on any stale lock. Run it after changing any
# Cargo.toml; scripts/gate.sh also checks the single workspace it is
# gating.
#
#   scripts/check-locks.sh          # verify only (CI-safe)
#   scripts/check-locks.sh --fix    # regenerate stale locks in place

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIX="${1:-}"
stale=0

for ws in "$ROOT"/components/*/; do
    [ -f "$ws/Cargo.toml" ] || continue
    if ! (cd "$ws" && cargo metadata --locked --format-version 1 >/dev/null 2>&1); then
        name="$(basename "$ws")"
        if [ "$FIX" = "--fix" ]; then
            echo "regenerating stale lock: $name"
            (cd "$ws" && "$ROOT/scripts/serial.sh" cargo metadata --format-version 1 >/dev/null)
        else
            echo "STALE LOCK: components/$name (run scripts/check-locks.sh --fix)"
            stale=1
        fi
    fi
done

if [ "$stale" -ne 0 ]; then
    exit 1
fi
echo "all workspace locks consistent"
