#!/usr/bin/env bash
set -euo pipefail

# Local dev server for the MLPL web REPL.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEB_DIR="$(dirname "$SCRIPT_DIR")/apps/mlpl-web"

PORT=9957

cd "$WEB_DIR"
exec trunk serve --port "$PORT" "$@"
