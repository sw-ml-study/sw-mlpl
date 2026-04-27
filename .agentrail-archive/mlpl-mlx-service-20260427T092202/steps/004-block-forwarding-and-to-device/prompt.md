Phase 2.5 (inserted): block forwarding +
to_device('cpu', x) materialization. The
peer-routing infrastructure landed in step 003
(peers module + --peer flag + Display impl +
workspace split); this step closes the loop by
wiring the orchestrator's evaluator to actually
forward `device("mlx") { ... }` blocks to a
registered peer.

Three pieces. Each builds on what step 003
landed:

1. **`PeerDispatcher` trait on `mlpl_eval::
   Environment`**. New trait with two methods:
   - `dispatch_block(&self, device: &str, source:
     &str, bindings: HashMap<String, DenseArray>)
     -> Option<Result<Value, EvalError>>`
     -- returns Some(...) when this dispatcher
     handles the device, None to fall through to
     the in-process feature path.
   - `fetch_tensor(&self, peer: &str, handle:
     &str) -> Result<DenseArray, EvalError>`
     -- materialize a peer-resident tensor.
   `Environment` gains an
   `Option<Arc<dyn PeerDispatcher>>` field +
   setter / getter. Default None.

2. **Hook the trait into the evaluator**.
   - `eval_device` in mlpl-eval/src/device.rs:
     before falling through to the in-process
     feature path, check if env has a
     PeerDispatcher AND the dispatcher returns
     Some(...) for this device. If so, render
     the body via `Display for Expr` to a source
     string + collect free-name Array bindings
     from env (substring-match on the rendered
     source for R1 simplicity; AST-level scope
     analysis is a follow-up). Forward via the
     dispatcher; bind the result locally as
     either `Value::DeviceTensor` (kind:tensor)
     or `Value::Array` from the source via
     `Value::Str` etc.
   - `to_device` in mlpl-eval (find current
     impl): when the second arg evaluates to a
     `Value::DeviceTensor` and the first is
     `"cpu"`, call `env.peer_dispatcher().fetch_
     tensor(peer, handle)` and rebind the result
     locally as Value::Array.

3. **Wire the trait in mlpl-serve**. New
   `crates/mlpl-serve/src/dispatcher.rs` module:
   `pub struct RemoteMlxDispatcher { peers:
   PeerRegistry, session_id: Uuid, token:
   String }` (or whatever shape captures the
   per-session auth context). `impl
   PeerDispatcher for RemoteMlxDispatcher` --
   POSTs to peer's `/v1/sessions/{id}/eval-on-
   device` with bindings encoded via
   `mlpl_mlx_serve::wire::encode_for_json`;
   POSTs to `/v1/sessions/{id}/transfer` for
   fetch_tensor; lifts errors. NOTE: this
   creates a path dep from mlpl-serve onto
   mlpl-mlx-serve for the wire format
   functions. Either:
   - Add the path dep in Cargo.toml (cleanest;
     mlpl-serve gains a thin tree).
   - Or re-export the wire format from a shared
     crate (more surgery for v1; defer to R3 if
     it grows).
   Pick the simpler path-dep route for this
   step.

   Install the dispatcher when creating a
   session: `mlpl_serve::sessions::create_
   session` either gains a peers parameter, or
   the eval handler installs the dispatcher
   into env before each eval call. Pick
   per-eval install for now (keeps create_
   session signature stable; cleared after
   each call for safety).

4. **Integration test** at
   `crates/mlpl-serve/tests/peer_routing_
   tests.rs`. Spin up BOTH `mlpl-serve`
   (orchestrator) AND `mlpl-mlx-serve` (peer)
   in-process on random localhost ports via
   their respective `build_app`-style entries.
   Cases:
   - `device("mlx") { iota(3) }` returns a
     Value::DeviceTensor with shape [3] and
     device "mlx".
   - `to_device("cpu", x)` round-trips an
     iota(3) tensor; result is the expected
     [0, 1, 2].
   - Strict-fault: an explicit CPU op on a
     DeviceTensor errors with the actionable
     message from step 002.
   - device("mlx") block with no peer
     registered errors cleanly (or falls back
     to in-process if the mlx feature is
     enabled at compile time; document the
     observed behavior).

5. Update `contracts/serve-contract/eval-on-
   device.md` with the orchestrator-side
   round-trip flow.

6. **Disk-aware** -- this step touches both
   workspaces. Build them independently. Run
   the new integration test in the main
   workspace; confirm services workspace tests
   still pass.

7. Quality gates + commit. References Saga R1
   step (block-forwarding insert).
