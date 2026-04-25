# PRD

MLPL is a Rust-first array and tensor programming language platform for machine learning, visualization, and experimentation.

## Goals
- build a clean-room Rust implementation of an array/tensor language
- make execution visually explainable
- support ML experimentation in Rust
- support compartmentalized implementation by multiple coding agents
- support deployment-flexible compute -- the same MLPL program should run with CPU on a laptop, MLX acceleration on an Apple Silicon host, and CUDA acceleration on a Linux + NVIDIA host, transparently and possibly across all three at once

## Primary outputs
- CLI / REPL (`mlpl-repl`, including `--connect <url>` thin client mode)
- Yew / WASM visual debugger (`apps/mlpl-web`)
- Long-running interpreter as a service (`mlpl-serve` REST API; ships in v0.17.0)
- Per-device service binaries -- `mlpl-mlx-serve` (Apple Silicon) and `mlpl-cuda-serve` (Linux + NVIDIA) -- proposed in `docs/refactor-services.md`, planned as Sagas R1 / R2 / R3
- Rust library crates (`mlpl-core`, `mlpl-array`, `mlpl-parser`, `mlpl-runtime`, `mlpl-eval`, `mlpl-autograd`, `mlpl-trace`, `mlpl-viz`, `mlpl-cli`, `mlpl-mlx`)
- demos and blog/video artifacts

## Deployment models

The platform supports three deployment shapes; each
MLPL program runs unchanged across them:

1. **Single-process local** -- `mlpl-repl` on a
   laptop, everything in-process. The default. CPU
   ops native; MLX ops via the in-process
   `mlpl-mlx` feature when on Apple Silicon.
2. **Single-host server** -- `mlpl-serve` running
   locally; clients (`mlpl-repl --connect`,
   browser, future ratatui / Emacs / desktop GUI)
   connect to it. Sessions persist across client
   restarts. Shipped in v0.17.0 (Saga 21).
3. **Multi-host services** -- planned in
   `docs/refactor-services.md` as Sagas R1 / R2.
   Orchestrator `mlpl-serve` on any host; MLX
   server on a Mac; CUDA server on a Linux host;
   any client can connect to the orchestrator and
   use `device("mlx") { ... }` / `device("cuda")
   { ... }` blocks that route over the LAN to the
   appropriate peer. Tensors stay on-device;
   blocks ship as program source; results return
   as opaque handles plus optional materialized
   values via `to_device("cpu", ...)`.
