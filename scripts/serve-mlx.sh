#!/usr/bin/env bash
set -euo pipefail

# Start the MLX connect server the right way on Apple Silicon:
#   - mlpl-serve built with --features mlx (so device("mlx") runs on the
#     Apple GPU and /v1/devices advertises "mlx"),
#   - bound to 0.0.0.0 for LAN access,
#   - serving the web UI from pages/,
#   - any prior instance stopped cleanly first.
#
# The Apple analog of serve-cuda.sh. mlx-rs only builds on macOS.
#
# Usage:
#   scripts/serve-mlx.sh            # start (build only if binary missing)
#   scripts/serve-mlx.sh --build    # force a fresh --features mlx build
#   MLPL_PORT=6464 scripts/serve-mlx.sh
#
# The build goes through serial.sh so it never deadlocks against another
# cargo/trunk on the shared target/.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/release/mlpl-serve"
PORT="${MLPL_PORT:-6464}"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "serve-mlx.sh: Apple Silicon (Darwin/arm64) only. On Linux use scripts/serve-cuda.sh." >&2
    exit 1
fi

if [ "${1:-}" = "--build" ] || [ ! -x "$BIN" ]; then
    echo "==> Building mlpl-serve --features mlx (release)..."
    ( cd "$ROOT/components/serve" && "$ROOT/scripts/serial.sh" \
        cargo build -p mlpl-serve --features mlx --release )
fi

echo "==> Stopping any running mlpl-serve..."
pkill -x mlpl-serve 2>/dev/null || true
sleep 1

echo "==> Starting mlpl-serve (mlx) on 0.0.0.0:$PORT ..."
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
echo "==> Open the web UI (auto-connects to this box as the MLX peer):"
for iface in en0 en1; do
    ip="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
    [ -n "$ip" ] && echo "    http://$ip:$PORT/sw-mlpl/?connect=http://$ip:$PORT"
done
