Phase 1 step 001: `crates/mlpl-serve` skeleton +
sessions + eval endpoint.

Build the minimal CLI server: a new binary+library
crate that exposes a REST API for "create a
session, eval programs against it." MVP only --
proxy, SSE, cancellation all stay deferred.

1. **New crate**.
   - `crates/mlpl-serve/Cargo.toml` -- binary AND
     library so tests can spin up the server in-
     process (`[[bin]]` + `[lib]` both pointed at
     the package).
   - Add to workspace `members` in root
     `Cargo.toml`.
   - Dependencies: `axum = "0.7"`, `tokio = {
     version = "1", features = ["full"] }`,
     `tower = "0.4"`, `serde = { version = "1",
     features = ["derive"] }`, `serde_json = "1"`,
     `uuid = { version = "1", features = ["v4"] }`,
     `rand = "0.8"`, `mlpl-parser = { path =
     "../mlpl-parser" }`, `mlpl-eval = { path =
     "../mlpl-eval" }`. Dev: `reqwest = {
     version = "0.12", features = ["json"] }`.
     Pin to currently-stable versions; bump only
     if a real conflict appears.

2. **Endpoints (MVP only)**.
   - `POST /v1/sessions` -- no auth needed (this
     IS how you get a token). Returns
     `{session_id: <uuid>, token: <random str>}`.
     Server creates an in-memory session entry
     containing a fresh `mlpl_eval::Environment`.
   - `POST /v1/sessions/{id}/eval` -- requires
     `Authorization: Bearer <token>` matching
     this session's token. Body:
     `{program: <string>}`. Lex + parse + run
     `eval_program_value` against the session's
     env. Returns
     `{value: <stringified>, kind: "array" |
     "string" | "model" | "tokenizer"}`. On
     `EvalError` return 400 with
     `{error: <message>}`.
   - `GET /v1/health` -- no auth.
     `{status: "ok", version: <crate version>}`.

3. **Session storage**.
   `tokio::sync::RwLock<HashMap<Uuid, Session>>`
   on the application state.
   `Session { token: String, env: mlpl_eval::
   Environment }`. In-memory only -- persistence
   is a future saga.

4. **Auth middleware**. Bearer extractor reads
   `Authorization`, looks up the session by id,
   compares tokens **constant-time**
   (`subtle::ConstantTimeEq` -- add `subtle =
   "2"` if needed). Mismatch / missing -> 401.
   Unknown session id -> 404.

5. **CLI flags**.
   - `--bind <host:port>` (default
     `127.0.0.1:6464`).
   - `--auth <required|disabled>` (default
     `required`). `--bind 0.0.0.0` REQUIRES
     `--auth required` -- refuse to start
     otherwise (print an error explaining why
     and exit non-zero).

6. **Module layout** (sw-checklist budget):
   - `crates/mlpl-serve/src/main.rs` (3-4 fns).
   - `crates/mlpl-serve/src/server.rs` -- router
     wiring + state + `pub fn run(addr,
     auth_mode) -> impl Future` for in-process
     test harness use (3-5 fns).
   - `crates/mlpl-serve/src/sessions.rs` --
     `Session` struct + creation + lookup + token
     generation (4-5 fns).
   - `crates/mlpl-serve/src/handlers.rs` -- one
     fn per route handler (3-5 fns).
   - `crates/mlpl-serve/src/auth.rs` -- bearer
     extractor + middleware (2-3 fns).
   - `crates/mlpl-serve/src/lib.rs` -- pub
     re-exports for the test harness.
   Each module under the 7-fn cap.

7. **Tests** at
   `crates/mlpl-serve/tests/api_tests.rs`. Spin
   the server up on a random localhost port via
   `server::run(...)`. Use `reqwest` to exercise:
   - `POST /v1/sessions` returns 200 with non-
     empty id + token.
   - `POST /v1/sessions/{id}/eval` with the right
     bearer token runs `iota(5)` and returns the
     array value as a string (or whatever the
     value-kind serialization produces).
   - `POST /v1/sessions/.../eval` no bearer ->
     401.
   - Wrong bearer token -> 401.
   - Unknown session id -> 404.
   - Eval error (e.g., `undefined_var`) -> 400
     with a message containing the offending
     name.
   - `GET /v1/health` -> 200, json status ok.
   - `mlpl_serve::server::run("0.0.0.0:0",
     AuthMode::Disabled)` returns an
     error / panics with a message about needing
     `--auth required` for non-loopback binds.

8. **Contract**: new
   `contracts/serve-contract/sessions-and-eval.md`
   -- session creation, eval endpoint shape,
   value-kind serialization, error catalog,
   security posture (constant-time compare,
   `--bind 0.0.0.0` requires `--auth required`),
   explicit non-goals (no LLM proxy, no SSE, no
   cancellation, no persistence, no
   `inspect` endpoint yet -- that lands in step
   002 alongside `--connect`).

9. Quality gates + commit. Commit message
   references Saga 21 step 001.
