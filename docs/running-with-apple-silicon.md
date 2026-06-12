# Running the connect server on Apple Silicon

A practical runbook for standing up the MLPL "connect" server
(`mlpl-serve --features mlx`) on an Apple Silicon Mac so the web
playground's `device("mlx") { ... }` blocks run on the Apple GPU
in-process, and `:ask` routes to a local Ollama.

For background see `docs/using-cli-server.md` (the server in general),
`docs/using-mlx.md` (the `device("mlx")` language feature), and
`docs/build-and-workspace-plan.md` (why the GPU compute lives in sibling
crates). This doc is the "how do I actually run it for a demo" version.

## TL;DR

```bash
# Build the MLX-aware server (Apple Silicon only)
cargo build -p mlpl-serve --features mlx --release   # from components/serve

# Local review on this Mac (no auth wall, loopback only)
./target/release/mlpl-serve --bind 127.0.0.1:6464 --auth disabled --static-dir pages

# Then open:
#   http://127.0.0.1:6464/sw-mlpl/?connect=http://127.0.0.1:6464
```

`/v1/devices` should report `{"devices":["cpu","mlx"]}`. If it only shows
`cpu`, the binary was not built with `--features mlx` (see Troubleshooting).

## What "connect" mode is

The web pages are pure WASM and contain no GPU code. GPU work always runs
server-side: the playground forwards `device("mlx") { ... }` blocks to a
peer server over HTTP, selected by the `?connect=<url>` query parameter.
On Apple Silicon that peer is an `mlpl-serve` built with `--features mlx`,
pointed at itself (it both serves the pages and is the MLX peer).

## Building

`mlx-rs` only builds on macOS, so the MLX server cannot be cross-built
from Linux -- build it on the Mac.

- From the serve workspace: `cargo build -p mlpl-serve --features mlx
  --release` (run inside `components/serve`).
- Or use `scripts/build-mlx.sh`, which builds the server AND the WASM
  pages and refuses to run off Apple Silicon. Only rebuild the pages when
  `apps/mlpl-web/` actually changed.

Notes verified in practice:

- The `mlx` feature uses the vendored `mlx-rs` with `default-features =
  false, features = ["accelerate"]` (no `metal`), so it compiles with the
  Xcode **Command Line Tools** -- a full `Xcode.app` is not required. See
  `vendor/mlx-rs/VENDORING.md`.
- The first build compiles `mlx-sys` (the MLX C++ library) and is slow;
  afterwards it is cached, so incremental `mlpl-serve --features mlx`
  builds finish in seconds.
- `mlpl-serve` is the only artifact you need for a demo; you do not need to
  rebuild `pages/` unless the web source changed.

### What `--features mlx` actually wires up

As of the GPU workspace split (S4), the MLX optimizer compute lives in the
sibling `mlpl-mlx-eval` crate, not in `mlpl-eval`. `--features mlx` pulls
that crate in, and the binary **registers** its `gpu_step()` at startup
(`mlpl-serve` `register_gpu_step()` in `main.rs`). Registration is
mandatory: a server without `--features mlx` advertises only `cpu`, and any
`device("mlx")` block silently falls back to the CPU tape.

## Running

```text
usage: mlpl-serve [--bind <host:port>] [--auth <required|disabled>]
                  [--peer <device>=<url>]... [--insecure-peers]
                  [--static-dir <path>]
Defaults: --bind 127.0.0.1:6464  --auth required
```

Two common modes:

| Goal | Command |
|------|---------|
| Local review on this Mac | `mlpl-serve --bind 127.0.0.1:6464 --auth disabled --static-dir pages` |
| LAN access from another device | `mlpl-serve --bind 0.0.0.0:6464 --auth required --static-dir pages` |

Rules to remember:

- **Non-loopback binds (`0.0.0.0:...`) require `--auth required`** -- the
  server refuses to expose the API to the network without auth.
- `--auth disabled` only works on a loopback bind and removes the token
  wall, which is what you want for a quick local browser review.
- `--static-dir pages` mounts the web UI at `/sw-mlpl/` on the same origin
  as the `/v1` API, so there is no CORS plumbing.

`scripts/serve-mlx.sh` is the maintained launcher for the LAN case: it
stops any prior instance, builds `--features mlx` on `--build` (or if the
binary is missing), binds `0.0.0.0:6464` with `--auth required`, serves
`pages/`, and prints the per-interface connect URLs. Override the port with
`MLPL_PORT=...`.

## The connect URL

Open the playground with a `?connect=` pointing at this server:

```text
http://<host>:<port>/sw-mlpl/?connect=http://<host>:<port>
```

- Local: `http://127.0.0.1:6464/sw-mlpl/?connect=http://127.0.0.1:6464`
- LAN: substitute the Mac's LAN IP (`ipconfig getifaddr en0`) for both
  `<host>` occurrences.

The `?connect=` value makes the playground auto-connect to this box as the
MLX peer, so MLX demos run on the Apple GPU here.

## Verifying it is up (and really on the GPU)

```bash
curl -s http://127.0.0.1:6464/v1/devices            # -> {"devices":["cpu","mlx"]}
curl -s -o /dev/null -w '%{http_code}\n' \
     http://127.0.0.1:6464/sw-mlpl/                 # -> 200 (web UI)
curl -s -o /dev/null -w '%{http_code}\n' \
     http://localhost:11434/api/tags                # -> 200 if Ollama is up
```

- `mlx` in `/v1/devices` confirms the feature is compiled in.
- The `:ask` demo needs a local **Ollama** on `localhost:11434`; start it
  separately if that returns non-200.
- A `device("mlx")` adam/LoRA block that has no GPU fast path (or an
  unregistered step) prints a one-time notice that it ran on the CPU --
  watch for that in the server log if a demo seems slow.

Logs go to wherever you redirect stdout/stderr (e.g.
`/tmp/mlpl-serve.log`).

## Stopping

```bash
pkill -x mlpl-serve
```

## Troubleshooting

- **`/v1/devices` shows only `cpu`.** The binary was built without
  `--features mlx`, or you are running an older binary. Rebuild with
  `--features mlx --release` and restart.
- **`device("mlx")` runs but is suspiciously fast / matches CPU exactly.**
  The GPU step may not be registered, so eval fell back to the CPU tape.
  Confirm the server was started from a `--features mlx` build; the binary
  registers `mlpl_mlx_eval::gpu_step()` at startup (the MLX parity test
  carries a non-bit-identical-to-CPU guard for exactly this reason).
- **"Apple Silicon (Darwin/arm64) only" from a script.** `build-mlx.sh` /
  `serve-mlx.sh` refuse to run off Apple Silicon; `mlx-rs` cannot be
  cross-built. Use `scripts/build-cuda.sh` on Linux+NVIDIA or
  `scripts/build-pages.sh` for a GPU-less host.
- **Server refuses to start on `0.0.0.0` without auth.** Add
  `--auth required` (or bind `127.0.0.1` and use `--auth disabled`).
- **`:ask` does nothing.** Start Ollama; verify `localhost:11434/api/tags`
  returns 200.
```
