Phase 2 step 003: orchestrator peer routing +
workspace split.

Wire `mlpl-serve` (the orchestrator from Saga 21)
to forward `device("mlx") { ... }` blocks to a
registered MLX peer running `mlpl-mlx-serve`.
Also split the MLX service into its own workspace
for the disk-pressure win the services refactor
was designed around.

1. **`--peer mlx=<url>` flag on `mlpl-serve`**.
   Add to its CLI parser (already hand-rolled in
   step 001 of Saga 21). Multiple `--peer`
   instances allowed (one per device). Refuses
   non-loopback URLs unless `--insecure-peers`
   is set (mirrors the existing `--bind 0.0.0.0`
   + `--auth required` precondition).

2. **Peer registry module**:
   `crates/mlpl-serve/src/peers.rs` (new). Stay
   under 7-fn budget.
   - `pub struct Peer { url: String, client:
     reqwest::blocking::Client, peer_token:
     Option<String> }` -- the token field is
     reserved for R3's per-peer auth; for R1
     loopback-only it stays `None`.
   - `pub type PeerRegistry =
     Arc<HashMap<String, Peer>>` (no RwLock --
     registry is built at startup from
     `--peer` flags and never mutated).
   - `pub fn parse_peer_arg(s: &str) ->
     Result<(String, String), String>` --
     parses `mlx=http://...` into
     `("mlx", "http://...")`.
   - `pub fn build_registry(peers: Vec<(String,
     String)>) -> PeerRegistry`.
   Threaded through `AppState`.

3. **Forward `device("<name>") { ... }`
   blocks** in mlpl-eval's evaluator. Today
   the evaluator's device-scope dispatch (in
   `crates/mlpl-eval/src/device.rs` or wherever
   `device::dispatched_call` lives) is purely
   in-process. Add a peer-routing hook BEFORE
   the in-process feature path:
   - If `state.peers` contains an entry for the
     requested device, marshal the block:
     - Extract the block source as a string
       (the parser must be able to surface the
       block body's source text -- if not, may
       need to re-render from Expr; document
       which path).
     - Walk the active `Environment` to find
       any free names referenced by the block;
       package those as `bindings` (encode
       arrays via `mlpl_mlx_serve::wire::
       encode_for_json`; skip non-Array
       values for R1 with a "block referenced
       <name> which is not a tensor; only
       Array bindings are forwarded in R1"
       error if a Model/Tokenizer is
       referenced).
     - POST to the peer's
       `/v1/eval-on-device` endpoint.
     - On `{kind: "tensor", ...}`: bind the
       result variable as a `Value::
       DeviceTensor`.
     - On `{kind: "string", ...}`: bind as
       `Value::Str`.
     - Errors propagate as `EvalError`.
   - If no peer for the device: fall through to
     the existing in-process feature path (no
     regression for single-host MLX users with
     `--features mlx`).

4. **`to_device("cpu", x)` fetch path**: when
   `x` is a `Value::DeviceTensor`, call the
   peer's `POST /v1/sessions/{id}/transfer`
   (added in step 002), decode the bytes via
   `decode_from_json`, rebind as
   `Value::Array`. Update
   `crates/mlpl-eval/src/device.rs` (or
   wherever `to_device` is implemented) with
   the remote-fetch path.

5. **Workspace split**: move
   `crates/mlpl-mlx-serve/` out to
   `services/mlpl-mlx-serve/` with its own
   `Cargo.toml` `[workspace]` declaration.
   - The new service workspace's `members`
     list contains just the binary itself.
   - It reaches back into the main workspace
     via path deps for `mlpl-mlx-rt`,
     `mlpl-eval`, `mlpl-parser`. Path deps
     ACROSS workspaces work but build into
     the consumer's `target/` -- which is the
     point.
   - Update root `Cargo.toml` to remove
     `crates/mlpl-mlx-serve` from `members`.
   - `crates/mlpl-mlx-rt/` STAYS in the main
     workspace -- it's the FFI shim that the
     interpreter uses for the in-process
     fallback.

6. **`scripts/build-mlx-serve.sh`**:
   convenience script: `cd services/mlpl-mlx-
   serve && cargo build --release`. Optional
   `--run` flag invokes the binary with the
   default loopback bind. Document in the
   contract + the using-mlx-service.md doc
   (which lands in step 004).

7. **Tests**:
   - `crates/mlpl-serve/tests/peer_routing_
     tests.rs` (new). Spin up BOTH
     mlpl-serve (orchestrator) AND
     mlpl-mlx-serve (peer) in-process on
     random localhost ports via their
     respective `build_app` library entries.
     Connect a session against the orchestrator;
     register the MLX peer via the test
     server's `--peer` config (or via a
     library setter on `AppState` for tests).
     Test cases:
     - `device("mlx") { x = iota(3) }` returns
       a `Value::DeviceTensor` with shape [3]
       and device "mlx".
     - `to_device("cpu", x)` round-trips the
       bytes; the resulting `Value::Array` has
       the expected contents.
     - `device("mlx") { ... }` with no
       registered peer falls back to in-
       process if the `mlx` feature is
       enabled at compile time, OR errors
       cleanly if neither peer nor feature
       is available.
     - Strict-fault: `device("cpu") { y = x +
       1 }` where `x` is a DeviceTensor
       errors with the expected actionable
       message.
   - `services/mlpl-mlx-serve/tests/`: keep
     the existing tests from step 002 but
     verify they still run after the
     workspace split (separate `target/`).

8. **Workspace split disk validation**: after
   the split, `cargo build` in the main
   workspace should NOT compile `mlpl-mlx-
   serve` (because it's no longer a member).
   Verify this; document the disk savings in
   the commit message (e.g., `du -sh target/`
   before vs after -- main workspace target
   should drop by ~3-5 GB once axum + tokio +
   reqwest are no longer compiled there for
   mlpl-mlx-serve's sake).
   Note: mlpl-serve still needs them.

9. **Disk-aware build hygiene**: this step
   touches both workspaces; build them
   independently. Don't run the WASM build
   (`scripts/build-pages.sh`) unless apps/
   mlpl-web/ actually changed (it shouldn't
   for R1).

10. Quality gates + commit. References Saga
    R1 step 003. Commit message documents the
    measured target/ size delta from the
    workspace split.
