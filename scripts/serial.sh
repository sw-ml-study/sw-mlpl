#!/usr/bin/env bash
set -euo pipefail

# Run a build/test command under a GLOBAL lock so two cargo (or trunk)
# invocations can NEVER run at the same time. The whole repo shares one
# target/ dir (.cargo/config.toml); concurrent cargo runs block on the
# same build lock and DEADLOCK -- producing no output while wall-clock
# burns. Routing every cargo/trunk call through this wrapper serializes
# them: a second caller WAITS for the first to finish, instead of
# deadlocking.
#
#   scripts/serial.sh cargo test -p mlpl-eval
#   scripts/serial.sh trunk build --release ...
#
# The lock is held only for the duration of THIS command (not across
# nested script calls), so a script may call serial.sh many times in
# sequence without self-deadlocking.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/target"
LOCK="$ROOT/target/.build.lock"

# flock -w: wait up to 30 min for the lock, then fail loudly rather than
# hang forever. exec replaces the shell so signals/exit codes pass through.
exec flock -w 1800 "$LOCK" "$@"
