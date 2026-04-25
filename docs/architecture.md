# Architecture

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
|  MLX ops via in-process     |
|   mlpl-mlx feature (Saga 14)|
|  CUDA in-process planned    |
|   (Saga 17, deferred)       |
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

## Design rules

- narrow public APIs
- contract-first development
- traceability as a first-class concern
- upstream-only visibility by default
- **device backends should be services, not features.** Once R1 lands, in-process device crates exist as fallbacks but the canonical path is the service. Reasoning: deployment flexibility (cross-host compute), disk pressure (separate target trees per service workspace), and hardware-coupling escape (one binary can't link both MLX and CUDA, but one orchestrator can route to peers that each link one).
