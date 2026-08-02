# Architecture

> **ERRATA NOTE (2026-08-02):** diagrams/sections below that label
> in-process CUDA as "planned/deferred (Saga 17)" are stale: an
> in-process CUDA feature and a CUDA-enabled `mlpl-serve` vertical
> slice are implemented (Linux/NVIDIA). The separate CUDA peer
> service, discovery, and multi-GPU/distributed operation remain
> future work. MLX is no longer eager per-op: saga E4 (landed
> 2026-08-02) replaced it with the device-resident `TensorHandle`
> tape described in the section below.

MLPL is organized as a cellular monorepo with narrow crates and matching contracts.

## Dependency flow (today)

`core -> array/parser -> runtime -> eval -> trace -> viz/wasm/apps -> ml`

Saga 21 (v0.17.0) added `mlpl-serve` (REST + bearer
auth + sessions + eval + inspect + health) sitting
on top of `eval`; the same dependency layering
holds.

## Service topology (today and proposed)

### Today: single-process or single-host server

```
+------------------------+
|  mlpl-repl (CLI)       |
|   or mlpl-web (WASM)   |
|   or mlpl-repl --connect (Saga 21)  |
+----------+-------------+
           |
           v
+-----------------------------+
|   mlpl-serve (Saga 21) OR   |
|   in-process mlpl-eval      |
|                             |
|  CPU ops native             |
|  MLX resident-tensor tape   |
|   via mlpl-mlx-handle (E4)  |
|  CUDA in-process vertical   |
|   slice (Linux/NVIDIA)      |
+-----------------------------+
```

### Proposed: device backends as services

See `docs/refactor-services.md` for the full design.
Promotes each device backend to its own long-running
service process so the orchestrator can route
device-scoped blocks to peers running on the right
hardware.

```
   [ mlpl-web ]   [ mlpl-repl --connect ]
            \      /
             v    v
        +---------------------+
        | mlpl-serve          |
        |  (orchestrator;     |
        |   CPU in-process;   |
        |   forwards device   |
        |   blocks to peers)  |
        +---+-------------+---+
            |             |
       +----v----+   +----v-----+
       | mlpl-   |   | mlpl-    |
       | mlx-    |   | cuda-    |
       | serve   |   | serve    |
       | (Apple) |   | (Linux)  |
       +---------+   +----------+
```

`device("mlx") { ... }` and `device("cuda") { ... }`
blocks ship as program-source payloads to the
appropriate peer; tensors stay on-device until an
explicit `to_device("cpu", ...)` materializes them
back. One round-trip per block, not per matmul.

The refactor is planned as three sequential sagas
(R1: refactor mlpl-mlx into mlpl-mlx-serve; R2:
CUDA-as-a-service replaces the deferred Saga 17;
R3: distributed primitives + auto-discovery).

## Device-resident tensor seam (saga E4, landed 2026-08-02)

`components/array/crates/mlpl-tensor-handle` is the wasm-clean
seam between the interpreter and device backends:

- `TensorHandle` is either a host `DenseArray` (bit-exact f64
  CPU reference) or an opaque `Dev` handle on a registered
  backend; `to_dense()` is the ONLY point that forces a lazy
  device graph.
- Backends self-register through a process-global `DeviceOps`
  OnceLock (the gpu_step inversion idiom); `mlpl-mlx-handle`
  (components/native-rt) is the MLX implementation over lazy
  `mlx_rs` arrays with Metal enabled.
- The autograd tape (`mlpl-autograd-tape` / `mlpl-autograd`)
  carries handles for node values AND gradients: forward,
  backward (including transpose/reshape and scalar-broadcast
  binaries), and the Adam/momentum optimizers all stay resident
  across `train` loops; only fused cross-entropy backward and a
  few structural kinds take the exact CPU kernels, re-joined by
  the mixed-residency accumulator.
- Seam counters (`uploads` / `downloads` / `submits` /
  `cpu_fallbacks` in `mlpl_tensor_handle::metrics`) make every
  boundary crossing observable; the bench prints them per step.
- The bespoke per-shape `gpu_step` fast paths are demoted below
  the resident path (E4 step 010): on MLX everything trains
  resident; `gpu_step` remains the CUDA route until the
  cuda-resident-tensors saga implements `DeviceOps` over Candle.

Numbers and the size-dependent MLX/CPU crossover:
`docs/benchmarks.md` ("Saga E4 step 008").

## Design rules

- narrow public APIs
- contract-first development
- traceability as a first-class concern
- upstream-only visibility by default
- **device backends should be services, not features.** Once R1 lands, in-process device crates exist as fallbacks but the canonical path is the service. Reasoning: deployment flexibility (cross-host compute), disk pressure (separate target trees per service workspace), and hardware-coupling escape (one binary can't link both MLX and CUDA, but one orchestrator can route to peers that each link one).
