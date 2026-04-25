# Refactor: Device Backends as Services

> **Status:** design / proposal. Not yet a saga. Treat
> as direction-setting until the first refactor saga
> opens. See `docs/plan.md` for the proposed saga
> decomposition.

## Why this exists

Today the MLX backend (`crates/mlpl-mlx`) is a Rust
library linked into the same process as the
interpreter. The planned CUDA backend (Saga 17) was
designed the same way: another optional feature on
the same single binary. Three problems with this:

1. **Hardware coupling.** A binary can't both link
   MLX (Apple-only, Accelerate / Metal) and CUDA
   (Linux + NVIDIA only). The Saga 14 design
   acknowledged this by making MLX optional via a
   Cargo feature, but you still build either an
   MLX-or-not binary -- there's no way to use both
   accelerators from one CLI session.
2. **Disk pressure.** Saga 21 (v0.17.0) just shipped
   `mlpl-serve`, which dragged axum + hyper + tokio
   + reqwest + rustls into the workspace. Together
   with the existing MLX bindings, the Yew web
   front-end, the autograd tape, and the Criterion
   bench harness, the workspace `target/` reaches
   30+ GB on a working dev tree -- enough to fill a
   constrained dev disk mid-session. The "dev host
   move to Linux" item exists in the plan partly to
   side-step this; refactoring also helps.
3. **Deployment inflexibility.** Today there is no
   way to keep MLX-accelerated training "available"
   from a non-Apple-Silicon dev host. The `device("
   mlx") { ... }` scope inside an .mlpl program
   silently no-ops on a non-MLX build (with a
   one-time warning). A user who develops on Linux
   but owns a Mac for MLX work loses the bridge.

## Today's architecture (in-process backends)

```
+--------------------------------------------+
|             mlpl-repl (CLI)                |
|  or mlpl-web (WASM) or mlpl-serve (REST)   |
+--------------------+-----------------------+
                     |  in-process calls
                     v
+--------------------------------------------+
|                mlpl-eval                   |
|  (interpreter; owns Environment + Tape)    |
+--+-------------------+-------------------+-+
   |                   |                   |
   v                   v                   v
+--------+        +---------+        +----------+
|mlpl-rt |        |mlpl-mlx |        |mlpl-cuda |
|(CPU)   |        |(Saga 14)|        |(Saga 17, |
|        |        |optional |        | planned, |
|        |        | feature)|        | optional)|
+--------+        +---------+        +----------+
```

The `device("mlx") { ... }` scoped form dispatches
ops inside the block through `mlpl-mlx` if the
feature is enabled, falling back to CPU otherwise.
Saga 17's plan was to add `device("cuda") { ... }`
the same way.

## Proposed architecture (device backends as services)

Promote each device backend to its own long-running
service process. The interpreter forwards
device-scoped blocks over the wire to the
appropriate peer, which holds device-resident
tensors and returns results.

```
   [ mlpl-web (WASM) ]    [ mlpl-repl (CLI) ]
              \                  /
               \   --connect    /
                \              /
            +--------------------+
            |  mlpl-serve (any   |
            |   host; CPU ops    |
            |   in-process; LLM  |
            |   proxy; routes    |
            |   device blocks)   |
            +---+------------+---+
                |            |
       device   |            |  device
        ("mlx") |            |  ("cuda")
          v     |            |     v
+----------------+      +-------------------+
| mlpl-serve     |      | mlpl-serve        |
| --device mlx   |      | --device cuda     |
| (Apple Silicon)|      | (Linux + NVIDIA)  |
+----------------+      +-------------------+
```

Three independent processes; any of them can be
running on any of three independent hosts. The CLI
or Web UI talks to whichever `mlpl-serve` is the
"orchestrator" (just the one they connect to).
That orchestrator forwards device-scoped blocks to
its registered peers when the local hardware can't
serve them.

### Key invariants

- **One server binary, multiple roles.** Same
  `mlpl-serve` binary; `--device <name>` flag (or
  feature-flag-gated set of devices it advertises)
  selects what hardware it manages locally.
  `--peer <name>=<url>` registers a peer that owns
  hardware this host doesn't.
- **Tensors live on-device.** A tensor created
  inside a `device("mlx") { ... }` block stays on
  the MLX peer's heap. The orchestrator gets back
  an opaque handle, not the data. `to_device("cpu",
  x)` is the explicit "ship me the bytes back"
  operation -- expensive, named so the user knows.
- **The CPU path stays in-process.** No
  network hop for CPU ops on the orchestrator's
  own session. Only `device("mlx") { ... }` and
  `device("cuda") { ... }` blocks forward.
- **Block-granularity RPC.** The whole scoped
  block ships as one program-source payload to
  the peer. The peer evaluates the block end-to-
  end on its hardware. One round-trip per block,
  not per matmul.
- **Session affinity.** Each session has one
  orchestrator. Peer connections are part of the
  orchestrator's state, not the client's. The
  client doesn't see the topology -- it just
  speaks to its session.

### Protocol additions to `mlpl-serve`

Layer over Saga 21's REST surface:

- `POST /v1/sessions/{id}/eval-on-device` --
  request body: `{device: "mlx" | "cuda", program:
  "<MLPL source>"}`. Server-side: forwards to the
  registered peer that owns `device`, awaits the
  result, returns `{value, kind, tensor_handles:
  [...]}`. If no peer owns the device, falls back
  to in-process (so a single-host MLX server still
  works without peer registration).
- `GET /v1/peers` -- lists registered peers + the
  devices each one advertises. No auth (LAN-only
  deployment assumption; LLM-proxy-style allow-
  lists are a separate concern).
- `POST /v1/peers/register` -- register a peer
  (or auto-discovery via mDNS for LAN; out of
  scope for the first refactor saga).
- `POST /v1/sessions/{id}/transfer` --
  `to_device(...)` materialization. Pull a peer-
  resident tensor handle back to the
  orchestrator's heap (or push a local tensor to
  a peer for subsequent device-scoped work).

The existing `POST /v1/sessions/{id}/eval`
remains unchanged for CPU work. Block-routing is
additive.

### Tensor lifecycle

Today every `Value::Array(DenseArray)` lives in
the orchestrator's `Environment`. After the
refactor:

```
Value::Array     -- CPU-resident (DenseArray bytes)
Value::DeviceTensor { peer, handle, shape, dtype }
                 -- peer-resident; opaque to the
                    orchestrator; freed on session
                    drop or explicit `release()`.
```

`device("mlx") { y = x * 2 }` translates to:
1. If `x` is a `Value::Array`: orchestrator pushes
   it to the MLX peer (one transfer), gets back a
   handle.
2. Orchestrator sends the block source + handle
   bindings to the peer.
3. Peer runs the block; resulting `y` is
   peer-resident; orchestrator gets a fresh
   handle.
4. Subsequent CPU ops on `y` either fault by
   default ("y lives on mlx; use to_device('cpu',
   y) to fetch") or auto-fetch with a warning.
   Pick the strict one for v1.

Reference counting + cleanup handled per-session;
session drop releases all peer handles owned by
that session.

## Worked example: LoRA fine-tune across Mac + Linux

```mlpl
# CLI on a Linux laptop, talking to an
# orchestrator (mlpl-serve on the same Linux
# host). MLX peer is a Mac at 192.168.1.10:6464,
# CUDA peer is a Linux host at 192.168.1.20:6464.
#
# Orchestrator was started with:
#   mlpl-serve --bind 127.0.0.1:6464 \
#              --peer mlx=http://192.168.1.10:6464 \
#              --peer cuda=http://192.168.1.20:6464

base = tiny_lm(seed=0)              # CPU on the
                                    # orchestrator
device("cuda") {                    # Block forwards
  base_cuda = to_device("cuda", base)  # to the
  pretrained = train(1000, ...)     # CUDA peer
}

device("mlx") {                     # Block forwards
  student = lora(pretrained, 8)     # to the MLX peer
  finetuned = train(50, ...)
}

device("cpu") {                     # back on the
  acc = evaluate(finetuned, test)   # orchestrator
}
```

Three peers, three networks of compute, one MLPL
program. The user only sees `device(...)`-scoped
forms; the routing is the orchestrator's job.

## Migration path / saga decomposition

This refactor is too big for a single saga. Propose
three sequential sagas, in this order:

### Saga R1 -- Refactor mlpl-mlx into mlpl-mlx-serve

Take the existing `crates/mlpl-mlx` and split it:

- `crates/mlpl-mlx-rt` -- the pure FFI surface
  (Accelerate / Metal bindings + ops). Library-
  only.
- `crates/mlpl-mlx-serve` -- a new binary that
  reuses `mlpl-serve`'s session + bearer-auth
  machinery and exposes a `--device mlx` mode.
  Implements the `eval-on-device` endpoint
  pattern by lex+parse+running the block source
  against an MLX-bound `Environment`.
- `mlpl-eval`'s `device::dispatched_call` keeps
  its in-process feature-gated path AS A
  FALLBACK, but gains a new "remote MLX peer"
  path that routes through the orchestrator's
  registered peers. Both paths remain valid;
  configuration picks one.

Disk benefit: `mlpl-mlx-rt` becomes the only
crate in the main workspace that imports the
heavy MLX bindings; `mlpl-mlx-serve` is its own
workspace target tree, separately buildable. A
Linux dev host that doesn't need MLX never
compiles either.

### Saga R2 -- CUDA backend AS A SERVICE (replaces Saga 17)

Saga 17 as currently planned (in-process CUDA
crate) gets retired. Replaced by:

- `crates/mlpl-cuda-rt` -- pure FFI for the CUDA
  + cuBLAS + cuDNN ops, paralleling
  `mlpl-mlx-rt`.
- `crates/mlpl-cuda-serve` -- binary; `--device
  cuda` mode of the same `mlpl-serve` shape.
- The orchestrator's peer-routing machinery is
  reused unchanged; the new peer just advertises
  `cuda` instead of `mlx`.

The dev-host move to Linux that already gates
Saga 17 still applies for building / testing
mlpl-cuda-rt natively. But the orchestrator and
the MLX peer can stay on the Mac.

### Saga R3 -- Distributed primitives + auto-discovery

Layered on R1 + R2:

- `run model on nodes[<n>]` syntax in MLPL --
  data-parallel training across registered CUDA
  peers.
- mDNS-style auto-discovery of peers on a LAN,
  so users don't have to hand-write peer URLs.
- Cross-peer tensor migration without going
  through the orchestrator's CPU heap (peer-to-
  peer transfer via direct HTTP).

Saga R3 is the original "Saga 17 distributed"
content; just renumbered + layered on the
service architecture.

## Disk-pressure secondary win

Splitting the workspace into separate target
trees is the single biggest disk win available
without dropping features. After R1 + R2, the
workspace topology becomes:

| Workspace | What it builds |
|---|---|
| `crates/` (main) | `mlpl-eval`, `mlpl-runtime`, `mlpl-autograd`, `mlpl-parser`, `mlpl-array`, `mlpl-trace`, `mlpl-viz`, `mlpl-cli`, `mlpl-serve`, `mlpl-mlx-rt` (FFI shim only), `mlpl-cuda-rt` (FFI shim only) |
| `services/mlpl-mlx-serve/` (new) | MLX device server binary + its dep tree (Accelerate / Metal bindings) |
| `services/mlpl-cuda-serve/` (new) | CUDA device server binary + its dep tree (cuBLAS, cuDNN) |
| `apps/mlpl-web/` (split out) | Yew + WASM front-end + its dep tree (`web-sys`, wasm-bindgen) |

Four separate `target/` trees. Each is buildable
independently. Mac dev never compiles CUDA;
Linux dev never compiles MLX; a Saga 11-only
language session (no web UI) never compiles
WASM artifacts.

The `.cargo/config.toml` per-workspace can also
set distinct `CARGO_TARGET_DIR` values pointing
at external storage if needed.

## Non-goals / open questions

- **Authentication between peers.** Current Saga
  21 bearer-token surface is per-session.
  Peer-to-peer auth needs its own design --
  either each peer gets a static API key, or a
  peer-trust certificate model. R1 is fine on
  loopback (`--bind 127.0.0.1`); LAN deployment
  needs the auth design first.
- **Fault tolerance.** What if the MLX peer
  goes down mid-block? Today the orchestrator
  would surface a network error; long-term the
  block could be retried, fail-fasted, or
  fallback-to-CPU. R3 territory.
- **Tensor serialization format.** Pickle is
  the Python world's answer; we have to pick.
  Saga 14's `to_device` already implements
  CPU<->MLX shape preservation; over-the-wire
  needs a versioned binary format (probably
  little-endian + length-prefixed-shape +
  raw bytes; `bincode` is a reasonable
  default). Open question for R1.
- **Auto-fetch on cross-device ops.** Strict-
  fault is the safe default; "auto-fetch with
  warning" is convenient but hides cost. Pick
  strict for v1.
- **Streaming + cancellation.** Saga 21 deferred
  these; they're more useful with peer routing
  (a long CUDA train block is a natural SSE
  stream). Likely lands in R3 or a separate
  Saga 21 follow-up.
- **`mlpl-repl --connect` to a non-orchestrator
  peer.** Probably should refuse -- a
  device-only peer doesn't host sessions.
  Document the role distinction.
- **Latency budget.** Block-granularity RPC
  amortizes over a whole train loop, but a
  single-matmul `device("cuda") { matmul(a, b)
  }` would be slow. Document the "use blocks,
  not single ops" pattern as the canonical idiom.

## Why not just use sccache / external CARGO_TARGET_DIR

`sccache` and external target dirs help with the
disk-pressure problem alone, but do not solve
the deployment / coupling / cross-host
compute story. The services refactor delivers
all three. Disk pressure relief is a free
secondary benefit.

If the user's pain is purely disk and not
deployment, `sccache` is a far simpler fix.
This refactor is justified by the deployment
story above.
