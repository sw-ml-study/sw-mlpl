# Saga R1: `mlpl-mlx` as a Service (v0.18.0)

> First saga of the services refactor proposed in
> `docs/refactor-services.md`. R2 (CUDA-as-a-service)
> follows once R1's protocol is settled; R3
> (distributed primitives + auto-discovery) layers on
> R1 + R2.

## Why this exists

Today `mlpl-mlx` is an optional Cargo feature on
`mlpl-eval`; `device("mlx") { ... }` scoped forms
dispatch in-process when the feature is enabled. R1
promotes the MLX backend to its own long-running
service binary so:

1. **An orchestrator on a Linux host can use MLX
   acceleration** by routing `device("mlx") { ... }`
   blocks to a `mlpl-mlx-serve` peer running on a
   Mac.
2. **Hardware coupling escapes** -- one binary still
   can't link both MLX and CUDA, but one
   orchestrator can route to peers that each link
   one. R2 (CUDA-as-a-service) reuses R1's
   block-routing machinery unchanged.
3. **Disk pressure relief.** `mlpl-mlx-serve` has
   its own service workspace with its own
   `target/`. The main workspace's `target/` no
   longer compiles MLX bindings.

R1 is the protocol-establishing saga. Once it
ships, R2 is mechanical (same shape, different
device).

## Decisions pinned (resolves design-doc open questions for R1)

The design doc flagged four open questions; for R1
specifically (single-host orchestrator + one MLX
peer, both on the user's LAN or loopback), pick
these defaults explicitly:

- **Peer auth: loopback-only for v1.** R1's
  `--peer mlx=<url>` flag refuses non-loopback URLs
  unless explicit `--insecure-peers` is passed
  (mirrors `mlpl-serve`'s `--bind 0.0.0.0`
  precondition). Per-peer bearer tokens land in R3
  with the LAN auto-discovery work.
- **Tensor wire format: `bincode` + explicit
  versioned envelope.** Format:
  `{version: u32 = 1, dtype: u8, ndim: u8, shape:
  [u64; ndim], data: [u8]}`. Endianness fixed
  little-endian. f64 only (matches MLPL's
  `DenseArray`); future dtype expansion bumps the
  version field.
- **Cross-device ops: strict-fault.** Touching a
  `Value::DeviceTensor` from a non-matching device
  scope errors with `"tensor lives on <peer>; use
  to_device('cpu', x) to fetch"`. No auto-fetch.
- **Streaming + cancellation: out of scope for
  R1.** Block-granularity RPC is sufficient for
  the MVP. SSE / cancellation lands as a Saga 21
  follow-up or in R3.

## Non-goals (deferred to R2 / R3 / Saga 21 follow-up)

- CUDA-as-a-service (Saga R2).
- Distributed primitives (`run model on nodes[...]`)
  (Saga R3).
- mDNS / LAN auto-discovery (R3).
- Peer-to-peer tensor migration without going
  through the orchestrator (R3 optimization).
- Streaming + cancellation (Saga 21 follow-up).
- `bf16` / `f16` tensor support (separate dtype
  saga; R1 wire format is f64-only with version
  field for future).
- Web UI re-routing to the new service mesh
  (separate frontend saga).

## Quality requirements (every step)

Identical to Saga 21 / 19 / 22. Design for sw-checklist
budgets up front. Disk-aware build hygiene from
`CLAUDE.md`'s "Disk-aware build hygiene" section
applies -- prefer scoped `cargo build -p <crate>`
during dev; full `cargo test` only at the end of a
step.

## Phase 1 -- crate refactor (1 step)

### Step 001 -- split mlpl-mlx into mlpl-mlx-rt + mlpl-mlx-serve crate skeleton

Pure refactor. No behavior change visible to MLPL
programs. The existing `device("mlx") { ... }`
dispatch path keeps working in-process via the
feature flag.

1. **Split `crates/mlpl-mlx`** into two crates:
   - `crates/mlpl-mlx-rt/` -- the FFI surface:
     Accelerate / Metal bindings, tensor ops, the
     existing `device::dispatched_call` integration
     point. Pure library; no async, no axum.
   - `crates/mlpl-mlx-serve/` -- new binary +
     library skeleton. Reuses
     `crates/mlpl-serve`'s session + auth +
     handler machinery either by direct path dep
     or by extracting a shared `mlpl-serve-core`
     crate (decide during impl based on which
     pieces are needed). For step 001, the
     binary just builds + serves `/health` --
     real eval routes land in step 002.
2. **Workspace topology**: keep both new crates
   in the main workspace for step 001 (simpler
   review, single `cargo test`). The disk-
   pressure-driven workspace split lands in step
   003 once the surface stabilizes.
3. **`mlpl-eval` dep update**: depend on
   `mlpl-mlx-rt` (not `mlpl-mlx`). The existing
   `mlx` feature flag still works -- just
   re-points at the renamed crate.
4. **No protocol yet**. `mlpl-mlx-serve` runs
   `/v1/health` (mirrors `mlpl-serve`) and a stub
   `POST /v1/eval-on-device` that returns 501 Not
   Implemented. Real implementation lands in
   step 002.
5. **Tests**: `cargo test -p mlpl-mlx-rt` (existing
   tests, just renamed paths). `cargo test -p
   mlpl-mlx-serve` (new: spin up the server,
   verify /health, verify /eval-on-device returns
   501).
6. **Contract**: new
   `contracts/serve-contract/eval-on-device.md`
   stub describing the endpoint shape that step
   002 fills in. Document the f64-only +
   versioned-envelope wire format decision.
7. Quality gates + commit. Commit message
   references Saga R1 step 001.

## Phase 2 -- tensor wire format + orchestrator routing (2 steps)

### Step 002 -- tensor wire format + Value::DeviceTensor

Add the cross-process tensor representation. No
orchestrator routing yet -- this step lands the
machinery, step 003 wires the orchestrator's peer
routing to it.

1. **Wire format module**:
   `crates/mlpl-mlx-serve/src/wire.rs` (or in a
   shared `mlpl-serve-core` if that gets
   extracted). Implements the bincode +
   versioned-envelope spec from the plan above.
   `encode_tensor(arr) -> Vec<u8>` and
   `decode_tensor(bytes) -> Result<DenseArray>`.
   Add `bincode` to mlpl-mlx-serve's deps.
2. **`POST /v1/eval-on-device`** real
   implementation in `mlpl-mlx-serve`. Body:
   `{program: <MLPL source>, bindings: [{name:
   <str>, tensor: <base64-bincoded>}]}`. The
   peer:
   - Decodes each binding into a `DenseArray`,
     binds it in a fresh `Environment`.
   - Runs the program with `device("mlx")` already
     set as the active device (so any nested
     `device("mlx")` no-ops; foreign device
     scopes still work via the in-process
     fallback feature).
   - Serializes the result tensor (or string)
     back into the response envelope.
3. **`Value::DeviceTensor` variant** in
   `crates/mlpl-eval/src/value.rs`:
   ```
   DeviceTensor {
       peer: String,        // peer URL
       handle: String,      // opaque token from
                            //   the peer
       shape: Vec<usize>,
       device: String,      // "mlx" / "cuda" / ...
   }
   ```
   `Display` impl prints
   `<tensor on <peer>:<device>; shape=[...]>`.
   `into_array` / `as_array` error with strict-
   fault message: `"tensor lives on <peer>; use
   to_device('cpu', x) to fetch"`.
4. **Peer-side handle store**: server-side
   `HashMap<String, DenseArray>` keyed by handle
   uuid; cleanup on session drop. Add
   `POST /v1/sessions/{id}/release` for explicit
   release before session end.
5. **Tests**: round-trip wire encode/decode for a
   handful of shapes (scalar, vector, matrix,
   rank-3); error case (invalid version field,
   shape/data length mismatch). `Value::
   DeviceTensor` `into_array` errors with the
   expected actionable message.
6. Quality gates + commit. References Saga R1
   step 002.

### Step 003 -- orchestrator peer routing + workspace split

Wire the orchestrator (`mlpl-serve`) to forward
`device("mlx") { ... }` blocks to a registered MLX
peer. Also split the MLX service into its own
workspace for the disk-pressure win.

1. **`--peer mlx=<url>` flag** on `mlpl-serve`
   `main.rs`. Refuses non-loopback URLs unless
   `--insecure-peers` is set. Multiple `--peer`
   flags allowed (one per device).
2. **Peer registry** in `crates/mlpl-serve/src/
   peers.rs` (new module). `HashMap<String, Peer>`
   keyed by device name. `Peer { url: String,
   client: reqwest::blocking::Client }`. Threaded
   through `AppState`.
3. **`device("mlx") { ... }` interception** in
   `mlpl-eval`'s evaluator. New
   `eval_device_block_remote` hook called BEFORE
   the in-process MLX feature path. If a peer for
   the requested device is registered, forward the
   block; else fall back to in-process. The
   in-process feature stays untouched -- additive
   change.
4. **`to_device("cpu", x)` fetch path**: when `x`
   is a `Value::DeviceTensor`, POST
   `/v1/sessions/{id}/transfer` to the peer to
   pull the bytes back; decode into a
   `DenseArray`; rebind locally.
5. **Workspace split**: move
   `crates/mlpl-mlx-rt/` to stay in the main
   workspace (FFI shim, small dep tree); move
   `crates/mlpl-mlx-serve/` out to
   `services/mlpl-mlx-serve/` with its own
   `Cargo.toml` workspace. Update root
   `Cargo.toml` `members` list. The new service
   workspace gets its own `target/` so the main
   workspace's tree doesn't carry MLX bindings.
6. **`scripts/build-mlx-serve.sh`**: convenience
   script to build + run `mlpl-mlx-serve` from
   the new workspace.
7. **Tests**: integration test in
   `crates/mlpl-serve/tests/peer_routing_tests.rs`
   that spins up both `mlpl-serve` (orchestrator)
   AND `mlpl-mlx-serve` (peer) in-process on
   random localhost ports; runs a
   `device("mlx") { x = randn([3,3]) }` block;
   asserts `x` came back as a `Value::
   DeviceTensor` with the expected shape; runs
   `to_device("cpu", x)` and asserts the bytes
   round-trip via the wire format.
8. Quality gates + commit. References Saga R1
   step 003.

## Phase 3 -- demo + docs + release (1 step)

### Step 004 -- demo + docs + release v0.18.0

1. **`demos/mlx_remote.mlpl`** CLI demo (similar
   shape to `demos/llm_tool.mlpl`): assumes a
   running `mlpl-mlx-serve` peer at
   `http://localhost:6465`; exercises a
   `device("mlx") { train ... }` block over the
   wire; prints the resulting loss curve.
   Header guard explains the prerequisites.
2. **`docs/using-mlx-service.md`** retrospective
   + user guide. Sections: status (shipped Saga
   R1 / v0.18.0); what this is about (cross-host
   MLX, hardware-coupling escape, disk pressure);
   `mlpl-mlx-serve` quickstart (build from the
   service workspace, start with `--bind`,
   verify `/health`); orchestrator
   `--peer mlx=<url>` walkthrough; tensor
   lifecycle (Value::DeviceTensor, strict-fault
   on cross-device, explicit `to_device('cpu',
   ...)`); migration from the in-process MLX
   feature (it still works as a fallback;
   `--peer mlx=<url>` takes precedence when
   set); the deferred non-goals list.
3. **`docs/configurations.md`** refresh: new
   row(s) for `mlpl-mlx-serve` and the
   orchestrator's `--peer` flag. Update the
   architecture diagram in the matrix.
4. **`docs/refactor-services.md`** status block
   updated: R1 marked SHIPPED; R2 / R3 still
   planned.
5. **Version bump**: `Cargo.toml`
   workspace.package.version `0.17.0 -> 0.18.0`
   in the main workspace AND in the new
   `services/mlpl-mlx-serve/Cargo.toml`
   workspace. Both move together for v1; later
   sagas can decouple if useful.
6. **`CHANGELOG.md`**: new v0.18.0 section
   documenting the refactor: Added
   (`mlpl-mlx-serve` binary, `--peer` flag,
   `Value::DeviceTensor`, wire format,
   `to_device` round-trip), Changed
   (`mlpl-mlx` -> `mlpl-mlx-rt`, in-process
   feature now a fallback), Tests, Scope notes
   (deferred to R2 / R3).
7. **`docs/saga.md`**: Saga R1 retrospective
   above Saga 21.
8. **`docs/status.md`**: R1 row Planned ->
   Completed.
9. `cargo build --release` (main workspace) +
   `cargo build --release` (services/mlpl-mlx-
   serve workspace).
10. `./scripts/build-pages.sh` if any
    `apps/mlpl-web/` source changed (probably
    not for R1).
11. `./scripts/gen-changes.sh`.
12. `/mw-cp` quality gates on both workspaces.
13. Tag `v0.18.0` locally; DO NOT push without
    explicit user confirmation.
14. `agentrail complete --done`.

## Dependency graph

```
001 mlpl-mlx-rt + mlpl-mlx-serve skeleton
        |
002 tensor wire format + Value::DeviceTensor
        |
003 orchestrator peer routing + workspace split
        |
004 demo + docs + release v0.18.0
```

Sequential; each step depends on the previous.

## After Saga R1

Saga R2 (CUDA-as-a-service) follows. R2 reuses
R1's wire format, peer registry, `Value::Device
Tensor`, and `--peer` flag machinery; adds the
`mlpl-cuda-rt` FFI crate + `mlpl-cuda-serve`
binary. The dev host move to Linux + NVIDIA gates
R2's testability but not R2's design.

R3 (distributed primitives + auto-discovery)
layers on R1 + R2. `run model on nodes[...]`
syntax, mDNS peer discovery, peer-to-peer tensor
migration without orchestrator round-trip.
