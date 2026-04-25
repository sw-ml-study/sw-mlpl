Phase 2 step 002: tensor wire format +
Value::DeviceTensor variant + real
`POST /v1/eval-on-device` implementation on
`mlpl-mlx-serve`.

Lands the cross-process tensor representation
end-to-end on the peer side. No orchestrator
routing yet -- step 003 wires the orchestrator's
peer registry to dispatch device-scoped blocks
through this endpoint.

1. **Wire format module**:
   `crates/mlpl-mlx-serve/src/wire.rs`. Implements
   the bincode + versioned-envelope spec. Add
   `bincode = "1"` to the crate's Cargo.toml.
   Functions:
   - `pub fn encode_tensor(arr: &DenseArray) ->
     Vec<u8>` -- serializes
     `{version: u32 = 1, dtype: u8 = 0 (f64),
     ndim: u8, shape: [u64; ndim], data: [u8]}`.
     Endianness fixed little-endian.
   - `pub fn decode_tensor(bytes: &[u8]) ->
     Result<DenseArray, WireError>` -- inverse;
     surfaces a typed error for invalid version,
     wrong dtype, shape/data length mismatch.
   - `pub fn encode_for_json(arr: &DenseArray) ->
     String` -- base64-of-bincode for the
     JSON-friendly transport. The endpoint body
     is JSON; tensor bytes go through base64.
   - `pub fn decode_from_json(s: &str) ->
     Result<DenseArray, WireError>`.
   Stay under 7-fn module budget.

2. **`POST /v1/eval-on-device`** real impl in
   `crates/mlpl-mlx-serve/src/handlers.rs`.
   Body shape (replaces the step 001 stub):
   ```json
   {
     "session_id": "<uuid>",
     "program": "<MLPL source>",
     "bindings": [
       {"name": "x", "tensor": "<base64-bincode>"},
       {"name": "y", "tensor": "<base64-bincode>"}
     ]
   }
   ```
   Behavior:
   - Look up the session (404 on unknown).
   - Bearer auth as usual (401 on missing/wrong).
   - For each binding, decode the tensor and bind
     it in a fresh sub-Environment forked from the
     session's Environment.
   - Push `device("mlx")` onto the env's device
     stack so any nested `device("mlx")` no-ops
     and other device scopes still dispatch via
     the in-process feature fallback.
   - Lex + parse + `eval_program_value` against
     the sub-Environment.
   - Walk the resulting Value:
     - `Value::Array(a)` -> stash on the peer's
       handle store under a fresh uuid handle.
       Return `{result: {kind: "tensor", handle:
       "<uuid>", shape: [..], device: "mlx"}}`.
     - `Value::Str(s)` -> return
       `{result: {kind: "string", value: <s>}}`.
     - Other variants (Model, Tokenizer) ->
       400 with "device blocks must return a
       tensor or string in R1; got <kind>".
   - On `EvalError`: 400 with the upstream
     message.

3. **Peer-side handle store**:
   `crates/mlpl-mlx-serve/src/handles.rs` (new
   module, 3-4 fns: `new_store`, `insert`,
   `get`, `release`). `Arc<RwLock<HashMap<Uuid,
   DenseArray>>>` threaded through `AppState`
   alongside the session map. Cleanup hook on
   session drop -- iterate handles owned by the
   session, remove them. Add
   `POST /v1/sessions/{id}/release-handle/{h}`
   for explicit early release; auth required.

4. **`Value::DeviceTensor` variant** in
   `crates/mlpl-eval/src/value.rs`:
   ```rust
   pub enum Value {
       Array(DenseArray),
       Str(String),
       Model(ModelSpec),
       Tokenizer(TokenizerSpec),
       /// Saga R1 step 002: peer-resident
       /// tensor reference. The tensor bytes
       /// live on the peer named by `peer`;
       /// the orchestrator only holds the
       /// handle + shape + device metadata.
       /// Attempting `into_array` /
       /// `as_array` on a DeviceTensor errors
       /// strict-fault; explicit
       /// `to_device('cpu', x)` is the only
       /// way to materialize the bytes back.
       DeviceTensor {
           peer: String,
           handle: String,
           shape: Vec<usize>,
           device: String,
       },
   }
   ```
   Update `into_array` + `as_array` impls to
   error with the strict-fault message:
   `"tensor lives on <peer>:<device>; use
   to_device('cpu', x) to fetch"`. Update the
   `Display` impl to print
   `<tensor on <peer>:<device>; shape=[..]>`.

5. **Peer-side `/v1/sessions/{id}/transfer`**
   endpoint (handler in mlpl-mlx-serve). Body
   `{handle: "<uuid>"}`. Auth required. Returns
   `{tensor: "<base64-bincode>"}` -- the bytes,
   ready for the orchestrator to decode and
   rebind locally as a `Value::Array`. Step 003
   wires the orchestrator's `to_device('cpu',
   ...)` path to call this.

6. **Tests**.
   - `crates/mlpl-mlx-serve/tests/wire_tests.rs`
     -- 6+ tests on encode/decode round-trip
     for scalar / vector / matrix / rank-3
     shapes; error cases (invalid version
     field, wrong dtype, shape/data length
     mismatch).
   - Extend
     `crates/mlpl-mlx-serve/tests/api_tests.rs`
     with the new endpoints:
     - eval-on-device happy path: pass a
       binding x = [1, 2, 3], program "x * 2",
       expect a tensor handle back; verify
       shape.
     - eval-on-device with no bindings, program
       "iota(5)" -> handle with shape [5].
     - transfer round-trip: eval to get a
       handle, transfer it, decode, assert
       equality with the expected output.
     - release-handle: insert a handle,
       release it, transfer it -> 404.
     - eval-on-device wrong session id -> 404,
       wrong bearer -> 401, malformed bindings
       -> 400.
   - Test the `Value::DeviceTensor`
     into_array strict-fault message in
     `crates/mlpl-eval/tests/value_tests.rs`
     (or wherever value tests live). One
     test: build a DeviceTensor, call
     `into_array()`, assert
     `EvalError::ExpectedArray` (or a new
     `EvalError::DeviceTensorFault` variant
     -- decide during impl which is more
     useful for downstream callers).

7. **Contract update**:
   `contracts/serve-contract/eval-on-device.md`
   gets the real endpoint shape (request +
   response), the wire format spec, the handle
   lifecycle (created on result, released on
   session drop or explicit release-handle),
   the strict-fault rule for cross-device
   ops in the orchestrator. Add a sibling
   `contracts/eval-contract/device-tensor.md`
   for the new Value variant.

8. **Disk-aware build hygiene**: scoped builds
   throughout. `cargo test -p mlpl-mlx-serve`
   for the wire + handle tests; `cargo test
   -p mlpl-eval` for the Value::DeviceTensor
   tests. Workspace-wide `cargo test` only at
   the end of the step.

9. Quality gates + commit. References Saga R1
   step 002.
