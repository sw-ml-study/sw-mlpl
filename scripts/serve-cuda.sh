#!/usr/bin/env bash
set -euo pipefail

# Start the CUDA connect server the right way, every time:
#   - mlpl-serve built with --features cuda (so device("cuda") runs on
#     the NVIDIA GPU and /v1/devices advertises "cuda"),
#   - CUDA_COMPUTE_CAP set (Blackwell/RTX-50xx = 120),
#   - bound to 0.0.0.0 for LAN access,
#   - serving the web UI from pages/,
#   - any prior instance stopped cleanly first.
#
# Usage:
#   scripts/serve-cuda.sh            # start (build only if binary missing)
#   scripts/serve-cuda.sh --build    # force a fresh --features cuda build
#   MLPL_PORT=6464 scripts/serve-cuda.sh
#
# The build goes through serial.sh so it never deadlocks against another
# cargo/trunk on the shared target/.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-120}"
BIN="$ROOT/target/release/mlpl-serve"
PORT="${MLPL_PORT:-6464}"

if [ "${1:-}" = "--build" ] || [ ! -x "$BIN" ]; then
    echo "==> Building mlpl-serve --features cuda (release)..."
    ( cd "$ROOT/components/serve" && "$ROOT/scripts/serial.sh" \
        cargo build -p mlpl-serve --features cuda --release )
fi

echo "==> Stopping any running mlpl-serve..."
pkill -x mlpl-serve 2>/dev/null || true
sleep 1

echo "==> Starting mlpl-serve (cuda) on 0.0.0.0:$PORT ..."
cd "$ROOT"
nohup "$BIN" --bind "0.0.0.0:$PORT" --auth required --static-dir pages \
    > /tmp/mlpl-serve.log 2>&1 &
sleep 2

pid="$(pgrep -x mlpl-serve || true)"
if [ -z "$pid" ]; then
    echo "FAILED to start mlpl-serve; last log lines:" >&2
    tail -10 /tmp/mlpl-serve.log >&2
    exit 1
fi

devices="$(curl -s -m 3 "http://127.0.0.1:$PORT/v1/devices" || echo '(no response)')"
ollama="$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://localhost:11434/api/tags 2>/dev/null || echo down)"
echo "==> mlpl-serve up (pid $pid)"
echo "    devices: $devices"
echo "    ollama:  $ollama (needed for the :ask demo)"
echo "    logs:    /tmp/mlpl-serve.log"
echo "==> Open the web UI (auto-connects to this box as the CUDA peer):"
for ip in $(ip -4 addr show scope global 2>/dev/null | grep -oP 'inet \K[\d.]+' | head -2); do
    echo "    http://$ip:$PORT/sw-mlpl/?connect=http://$ip:$PORT"
done
