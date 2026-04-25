Phase 1 step 001: split mlpl-mlx into mlpl-mlx-rt
(FFI library) + mlpl-mlx-serve (binary skeleton).
Pure refactor; NO behavior change visible to MLPL
programs. The existing `device("mlx") { ... }`
in-process dispatch keeps working through the
feature flag.

This step lands the crate skeleton + the
contract stub. Real wire-format encoding lands in
step 002; orchestrator peer routing lands in step
003.

1. **Split `crates/mlpl-mlx`** into two crates:
   - `crates/mlpl-mlx-rt/` -- pure FFI library.
     Contains everything from today's
     `crates/mlpl-mlx/src/`. The `device::
     dispatched_call` integration point in
     `mlpl-eval` should keep working unchanged
     after the rename.
   - `crates/mlpl-mlx-serve/` -- new package
     that is BOTH a library and a binary
     (matches `crates/mlpl-serve` shape).
     Library exposes a `pub fn run(addr,
     auth_mode) -> impl Future` for in-process
     test harness use. Binary `main.rs` is a
     thin shell over `run`.

2. **Workspace topology**: BOTH new crates stay
   in the main workspace for step 001 (simpler
   review, single `cargo test` covers both). The
   workspace split into a separate `services/`
   tree lands in step 003 once the surface is
   stable. Update the root `Cargo.toml`
   `members` list to add the two new crate dirs;
   remove `crates/mlpl-mlx`.

3. **`mlpl-mlx-serve` deps** (in its
   `Cargo.toml`):
   - `mlpl-mlx-rt = { path = "../mlpl-mlx-rt" }`
   - `mlpl-eval = { path = "../mlpl-eval" }`
     (server-side eval against an MLX-bound
     Environment)
   - `mlpl-parser = { path = "../mlpl-parser" }`
   - `axum = "0.7"`, `tokio = { version = "1",
     features = ["full"] }`, `tower = "0.4"`,
     `serde`, `serde_json`, `uuid`, `rand`,
     `subtle = "2"` (matches mlpl-serve's
     dependency set; auth + sessions
     machinery).
   - **NOTE**: rather than copy-pasting
     mlpl-serve's session/auth/handlers code
     into mlpl-mlx-serve, prefer one of:
     (a) a lightweight `pub` re-export from
     `mlpl-serve`'s lib.rs of the
     reusable pieces (`AuthMode`,
     `extract_bearer`, `check_token`, the
     session map type, `build_app`-style
     constructors); or
     (b) extract a shared `crates/mlpl-serve-
     core/` crate. Pick (a) for minimum
     disruption in step 001; revisit in step
     003 if the duplication grows. Either way,
     do NOT duplicate the auth code.

4. **Endpoints** for step 001 (real
   implementations land in step 002):
   - `GET /v1/health` -- mirror mlpl-serve's
     contract. Returns `{status: "ok",
     version: <crate version>, device:
     "mlx"}`.
   - `POST /v1/sessions` -- mirror
     mlpl-serve's session creation. The peer
     holds an MLX-bound Environment per
     session.
   - `POST /v1/eval-on-device` -- STUB.
     Returns `501 Not Implemented` with body
     `{error: "wire format ships in Saga R1
     step 002"}`. Real impl in step 002.
   The `--bind 0.0.0.0` + `--auth required`
   precondition from mlpl-serve carries over.

5. **`mlpl-eval` dep update**: change the
   optional `mlpl-mlx` path dep to
   `mlpl-mlx-rt`. The `mlx` Cargo feature still
   exists and still gates in-process dispatch
   through `device::dispatched_call`; just the
   underlying crate name changes. Update any
   `use mlpl_mlx::...` paths to `use
   mlpl_mlx_rt::...`.

6. **Module layout for mlpl-mlx-serve**
   (sw-checklist budget):
   - `crates/mlpl-mlx-serve/src/main.rs` --
     arg parsing + tokio runtime + run
     orchestration (3-4 fns).
   - `crates/mlpl-mlx-serve/src/lib.rs` --
     pub re-exports for the test harness.
   - `crates/mlpl-mlx-serve/src/server.rs` --
     `build_app`, `run`, `AppState`,
     `ServerError`. Reuses mlpl-serve's
     AuthMode + auth helpers via path import
     (or pub re-export -- see step (3)
     above). 3-5 fns.
   - `crates/mlpl-mlx-serve/src/handlers.rs`
     -- `health_handler`,
     `create_session_handler`,
     `eval_on_device_stub`. 3 fns.
   Stay under the 7-fn cap per module.

7. **Tests** at
   `crates/mlpl-mlx-serve/tests/api_tests.rs`.
   - `GET /v1/health` returns 200 with `device:
     "mlx"`.
   - `POST /v1/sessions` returns 200 with
     non-empty id + token.
   - `POST /v1/eval-on-device` returns 501
     with the expected stub message.
   - The same `run("0.0.0.0:0",
     AuthMode::Disabled)` insecure-bind safety
     check that mlpl-serve has.

8. **Contract**: new
   `contracts/serve-contract/eval-on-device.md`.
   Document the endpoint shape that step 002
   will implement (the request body's
   `program` + `bindings` fields, the `bincode`
   + versioned-envelope tensor wire format with
   the f64-only constraint and the version
   field for future dtype expansion). Mark the
   endpoint as "stub returning 501 in step 001;
   real impl ships in step 002" with a
   forward-pointer.

9. **Disk-aware build hygiene** (per CLAUDE.md):
   prefer `cargo build -p mlpl-mlx-rt -p
   mlpl-mlx-serve` during dev; full `cargo
   test` only at the end. After the step, if
   `target/` exceeds 10 GB, run `cargo clean`.

10. Quality gates + commit. Commit message
    references Saga R1 step 001 and notes that
    the refactor is pure-rename in step 001;
    behavior changes (wire format, peer
    routing) come in steps 002 + 003.
