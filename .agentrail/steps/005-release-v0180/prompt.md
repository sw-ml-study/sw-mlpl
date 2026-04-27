Phase 3 step 004: demo + docs + release v0.18.0.

1. **`demos/mlx_remote.mlpl`** CLI demo. Same
   shape as `demos/llm_tool.mlpl` (header guard
   explaining prerequisites). Sketch:
   - Header: requires a running mlpl-mlx-serve
     peer at http://localhost:6465 AND an
     orchestrator with `--peer
     mlx=http://localhost:6465`. Skip on CI /
     no-MLX envs.
   - Tour 1: simple `device("mlx") { x =
     iota(5) }` returns a Value::DeviceTensor;
     print to confirm the strict-fault display
     format.
   - Tour 2: `to_device("cpu", x)`
     materializes bytes back; print the array.
   - Tour 3: a small training block --
     `device("mlx") { m = chain(linear(8, 16,
     0), linear(16, 8, 1)); y = adam(loss, m,
     ...) }` -- prints the resulting loss
     curve. Verify against a running peer
     during impl.
   - Run-guide line at bottom:
     `mlpl-serve --bind 127.0.0.1:6464 --peer
     mlx=http://localhost:6465` (orchestrator)
     plus `cargo run --release --manifest-path
     services/mlpl-mlx-serve/Cargo.toml --
     --bind 127.0.0.1:6465 --auth disabled`
     (peer; loopback + auth disabled is fine
     for the demo).

2. **`docs/using-mlx-service.md`** retrospective
   + user guide. Sections:
   - Status (shipped Saga R1 / v0.18.0).
   - What this is about (cross-host MLX,
     hardware-coupling escape so R2 can land
     CUDA-as-a-service the same way, disk-
     pressure relief from the workspace split).
   - mlpl-mlx-serve quickstart: build from
     services/mlpl-mlx-serve workspace, start
     the binary, verify /health.
   - Orchestrator `--peer mlx=<url>`
     walkthrough: how to register the peer at
     orchestrator startup; refusal of non-
     loopback URLs without `--insecure-peers`.
   - Tensor lifecycle: Value::DeviceTensor +
     strict-fault on cross-device + explicit
     to_device('cpu', ...) materialization.
   - Wire format (high-level): bincode +
     versioned envelope; f64-only in R1; future
     dtype expansion via the version field.
   - Migration from the in-process MLX feature:
     in-process still works as a fallback when
     no peer is registered; --peer takes
     precedence when set. Single-host MLX users
     with `--features mlx` see no behavior
     change.
   - Multi-client picture: any client (CLI
     local, --connect, future ratatui / Emacs
     / web via R3) sees the orchestrator;
     only the orchestrator knows about peers.
   - Deferred non-goals: list (R2 CUDA-as-a-
     service, R3 distributed primitives + LAN
     auto-discovery + peer-to-peer tensor
     migration, peer-to-peer auth beyond
     loopback, streaming + cancellation,
     bf16/f16 dtypes, web UI re-routing).

3. **`docs/configurations.md`** refresh. Add
   row(s) for mlpl-mlx-serve and the `--peer`
   flag. Update the architecture diagram
   description to note the orchestrator + peer
   topology. Update existing footnotes [3] /
   [7] / [8] if they reference the in-process
   MLX dispatch as the primary path -- after
   R1 the canonical path is the service.

4. **`docs/refactor-services.md`** status block
   updated: R1 marked SHIPPED with the v0.18.0
   tag. R2 / R3 still planned. Add a brief
   "lessons from R1" subsection summarizing
   what worked + what to refine in R2.

5. **Version bump**: `Cargo.toml`
   workspace.package.version `0.17.0 -> 0.18.0`
   in BOTH workspaces (main + services/mlpl-
   mlx-serve). Both move together for v1.

6. **`CHANGELOG.md`**: new v0.18.0 section
   above v0.17.0. Sections (matches the prior
   release templates):
   - Added: `mlpl-mlx-serve` binary +
     library; `--peer mlx=<url>` flag on
     mlpl-serve; Value::DeviceTensor variant;
     `bincode` + versioned-envelope tensor
     wire format; orchestrator peer routing
     for `device("mlx") { ... }` blocks;
     `to_device('cpu', x)` round-trip via the
     transfer endpoint; demos/mlx_remote.mlpl;
     docs/using-mlx-service.md.
   - Changed: `crates/mlpl-mlx` ->
     `crates/mlpl-mlx-rt` (FFI shim only); the
     in-process MLX Cargo feature is now a
     fallback that activates only when no
     `--peer mlx=...` is registered. Workspace
     split: `mlpl-mlx-serve` moved to
     `services/mlpl-mlx-serve/` with its own
     workspace target tree.
   - Tests: list counts (wire encode/decode,
     api_tests on the peer, peer_routing_
     tests on the orchestrator, value-tensor
     fault tests).
   - Scope notes: R2 (CUDA-as-a-service), R3
     (distributed primitives + LAN auto-
     discovery + peer-to-peer migration),
     bf16/f16 dtypes, peer-to-peer auth
     beyond loopback, streaming + cancellation
     all explicit non-goals deferred to
     follow-up sagas.

7. **`docs/saga.md`**: Saga R1 retrospective
   above Saga 21 (newest-first ordering
   preserved). Cover the design deviations
   that surfaced during impl, the disk-savings
   measurement from the workspace split, the
   open questions that R2 inherits.

8. **`docs/status.md`**: R1 row Planned ->
   Completed; intended-sequence row for R2
   updated to "next, dev host move to Linux
   gates testing but not design"; Next-saga
   pointer rewrites to point at R2.

9. `cargo build --release` in BOTH workspaces.

10. `./scripts/build-pages.sh` ONLY if apps/
    mlpl-web/ source changed (probably not for
    R1; skip if so to save the WASM build
    time + disk).

11. `./scripts/gen-changes.sh` to refresh
    CHANGES.md in the main workspace; commit.

12. `/mw-cp` quality gates on both workspaces.
    For the services workspace, ensure its
    own `target/` size doesn't exceed the
    10 GB threshold; if it does, document
    why or trim deps.

13. Tag `v0.18.0` locally; DO NOT push without
    explicit user confirmation (cadence per
    prior releases).

14. `agentrail complete --done`.
