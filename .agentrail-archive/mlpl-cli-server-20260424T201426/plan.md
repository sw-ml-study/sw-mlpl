# Saga 21: CLI Server + Multi-Client UI (v0.17.0)

## Why this exists

Today MLPL has two surfaces: the local CLI REPL
(`mlpl-repl`) and a browser-only WASM REPL
(`apps/mlpl-web`). They share the language but
share nothing else -- different runtimes, different
visualization paths, no way for the browser to
reach a localhost LLM, no way for `mlpl-repl` to
attach to a long-running session on another host.

Saga 21 builds the missing piece: a long-running
MLPL interpreter exposed as a REST + WebSocket
server (`crates/mlpl-serve`), with multiple thin
clients connecting to it. One server, many clients.

The headline use cases:

- **Browser unblocking.** Saga 19's `llm_call`
  (and the `:ask` REPL command) cannot reach a
  localhost Ollama from the browser without CORS
  allow-listing + a server-side proxy. The web UI
  served by `mlpl-serve` is same-origin with the
  REST API, so the browser POSTs to the server,
  and the server (with explicit allow-list) does
  the real Ollama call. Same-origin, no CORS
  gymnastics on the client. (Proxy is post-MVP --
  the server skeleton lands first.)
- **Linux move continuity.** Once dev moves off
  Apple Silicon the local browser demo loses MLX.
  `mlpl-serve` running on the Apple-Silicon host
  with `--features mlx` stays the path: any
  client (web, CLI, future TUI/Emacs) gets MLX
  acceleration through the server.
- **mlpl-repl --connect.** Lets a CLI user attach
  to a remote session -- carry workspace state
  across machines, share a session with another
  user, work on an iPad through a thin client.
- **CLI visualization fix.** Today
  `mlpl-repl` prints raw `<svg>...</svg>` XML when
  a viz primitive runs interactively. The CLI viz
  strategy (auto-write to a cache dir, print the
  path) ships as part of this saga because the
  same machinery serves both the local CLI and
  `--connect` mode.

CLI server is prioritized **before** the dev-host
move to Linux because once dev is off Apple
Silicon the live browser demo loses local MLX.
Saga 17 (CUDA + distributed) is the *next* saga
after Saga 21 + the host move.

## Non-goals (deferred)

This saga ships the MVP. Several items the design
brief calls out are explicitly **post-MVP** and
land in follow-up sagas after the server stabilizes:

- **Server-side LLM proxy with allow-list.** The
  endpoint that lets the browser call
  `llm_call(...)` against a server-side
  allow-listed Ollama. Requires a careful security
  review (allow-list config, env-var secrets, rate
  limiting). Lands in a follow-up after the
  basic server skeleton is proven.
- **Visualization storage URLs.** Today's CLI
  viz strategy writes SVGs to a local cache dir
  with a printed path. The server-mode equivalent
  would mint URLs the client can fetch back. Out
  of scope for MVP; the CLI cache-dir story is
  enough for the local case.
- **Server-Sent Events / streaming eval.** Today
  every `eval` is a single request/response. SSE
  for partial output (loss curves during a
  `train { }` loop, streaming `llm_call`) is a
  natural follow-up but requires a different
  return-shape contract than the MVP.
- **Cancellation.** The MVP eval endpoint blocks
  until the program finishes. A cancel endpoint
  + cooperative interrupt point in `eval_program`
  are useful but not load-bearing for MVP.
- **Desktop GUI wrapper (tauri / wry).** A
  separate saga once the server contract is
  stable.
- **Emacs client.** Same -- depends on a stable
  server + a streaming SVG/PNG render path. Out
  of scope.
- **ratatui TUI client.** Same. CLI-only fast
  path is `mlpl-repl --connect`.
- **Web UI re-routing to call origin.** Today
  the web UI runs entirely in WASM in the
  browser. Pointing it at `mlpl-serve` instead
  is a non-trivial change to `apps/mlpl-web` and
  is worth its own scope; the MVP just exposes
  the API.

## Quality requirements (every step)

Identical to Sagas 19 / 22. `docs/sw-checklist-
patterns.md` is the decomposition reference.
Design for budgets up front. Every step ends with
`/mw-cp` clean (cargo test, clippy --all-targets
--all-features -- -D warnings, fmt, markdown-
checker on changed files, sw-checklist baseline
held).

## Phase 1 -- server skeleton (1 step)

### Step 001 -- `crates/mlpl-serve` skeleton + sessions + eval endpoint

1. **New crate**. `crates/mlpl-serve` -- a binary
   crate (also a library, so the test harness can
   spin up the server in-process). Add to the
   workspace `members` in the root `Cargo.toml`.
2. **HTTP framework**. `axum` 0.7+ on top of
   `tokio`. Modern, well-supported, simple
   layered router. `axum-extra` if needed for
   typed bearer-auth extractor.
3. **Endpoints (MVP)**.
   - `POST /v1/sessions` -> `{session_id, token}`.
     Creates an in-memory session entry containing
     a fresh `mlpl_eval::Environment`.
     Generates a UUIDv4 session id and a
     cryptographically random bearer token.
   - `POST /v1/sessions/{id}/eval` -- requires
     `Authorization: Bearer <token>` matching the
     session. Body: `{program: string}`. Runs
     `eval_program_value` against the session's
     env. Returns `{value: <serialized>, kind:
     "array" | "string" | ...}`. On `EvalError`
     return `400` with `{error: <message>}`.
   - `GET /v1/health` -> `{status: "ok",
     version: "<crate version>"}`. No auth.
4. **Session storage**. `tokio::sync::RwLock<
   HashMap<Uuid, Session>>` on the application
   state. `Session { token: String, env:
   Environment }`. In-memory only; persistence is
   a future saga.
5. **Auth middleware**. Extracts the bearer token
   from `Authorization`, looks up the session by
   id, compares constant-time. Reject with `401`
   on mismatch / missing.
6. **CLI flags**.
   - `--bind <host:port>` (default
     `127.0.0.1:6464`).
   - `--auth <required|disabled>` (default
     `required`). `--bind 0.0.0.0` REQUIRES
     `--auth required`; refuse to start
     otherwise.
7. **Module layout** (sw-checklist budget design):
   - `crates/mlpl-serve/src/main.rs` -- arg parse
     + tokio runtime + server bootstrap (3-4 fns).
   - `crates/mlpl-serve/src/server.rs` -- router
     wiring + state setup + `pub fn run(addr,
     auth_required) -> impl Future` for in-process
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
8. **Tests**. Integration tests at
   `crates/mlpl-serve/tests/api_tests.rs`. Use
   `reqwest` (or axum's TestClient via
   `tower::ServiceExt`) to exercise:
   - `POST /v1/sessions` returns 200 with an id +
     token; both non-empty.
   - `POST /v1/sessions/{id}/eval` with the right
     token runs `iota(5)` and returns the array.
   - Missing auth -> 401.
   - Wrong token -> 401.
   - Unknown session id -> 404.
   - Eval error (e.g., undefined variable)
     -> 400 with the message.
   - Health endpoint returns ok.
   - `--bind 0.0.0.0` with `--auth disabled`
     refuses to start.
9. **Contract**. New
   `contracts/serve-contract/sessions-and-eval.md`
   -- session creation, eval endpoint shape,
   error catalog, security posture (constant-time
   token compare, `--bind 0.0.0.0` requires
   `--auth required`), non-goals (no LLM proxy,
   no SSE, no cancellation).

## Phase 2 -- `mlpl-repl --connect` client (1 step)

### Step 002 -- `mlpl-repl --connect <url>`

1. **Flag**. New `--connect <url>` flag on
   `mlpl-repl`. Cannot combine with `-f` or
   `--data-dir` (those are local-mode only); error
   helpfully if combined.
2. **Behavior**. With `--connect <url>` set, the
   REPL:
   - On startup: POST `<url>/v1/sessions`, store
     the returned `(session_id, token)`.
   - For each user line: POST
     `<url>/v1/sessions/{id}/eval` with the line
     as `{program}`, print the returned
     `value` (or the error message). The local
     `Environment` is unused -- the server holds
     all state.
   - Slash commands (`:vars`, `:models`,
     `:experiments`, `:describe`, `:ask`, etc.)
     stay client-side BUT operate on a
     server-fetched snapshot. **MVP scope:**
     introduce a single inspector endpoint
     `GET /v1/sessions/{id}/inspect` that returns
     a JSON workspace summary (var names + shapes
     + `[param]` tags + model names + experiment
     names); the client renders it locally. This
     keeps the MVP small without breaking
     existing slash commands.
3. **Module**. New
   `apps/mlpl-repl/src/connect.rs`. Functions:
   `create_session`, `eval_remote`,
   `inspect_remote`, `read_loop`. Stay under
   7-fn budget.
4. **Tests**. Integration test at
   `apps/mlpl-repl/tests/connect_tests.rs` that
   spins up `mlpl-serve` in-process via the
   `lib.rs` `run(...)` entry point, runs a few
   evals through `--connect`, asserts results.
5. **Contract update**. Append a `Connect mode`
   section to a new
   `contracts/repl-contract/connect.md` (or to
   an existing repl contract if there is one) --
   document the slash-command snapshot pattern
   and the local-mode-only flags.
6. **`mlpl-serve`** gets the `GET /v1/sessions/
   {id}/inspect` endpoint added (small extension,
   shipped as part of step 002 since it's the
   client's primary slash-command source).
   Update the contract.

## Phase 3 -- CLI visualization strategy + docs (1 step)

### Step 003 -- CLI viz cache dir + `docs/using-cli-server.md`

1. **CLI viz cache strategy**. Today
   `mlpl-repl` (and `--connect`) prints raw
   `<svg>...</svg>` XML inline whenever a viz
   primitive returns. Replace that with:
   - A new `mlpl-cli` helper that detects when a
     `Value::Str` starts with `<svg` (or any other
     known viz format -- start with SVG; PNG /
     other deferred).
   - Writes the content to
     `$MLPL_CACHE_DIR/<sha256-prefix>.svg`
     (default `~/.cache/mlpl/`; respect
     `MLPL_CACHE_DIR` env var override).
   - Prints `viz: <path>` instead of the raw
     XML.
   - Returns the path as the new string value
     (so downstream programs get the path, not
     the XML -- discuss in the contract whether
     this is the right semantics; lean toward
     "yes, the user can `cat $path` if they want
     XML" for ergonomics).
2. **Module**.
   `crates/mlpl-cli/src/viz_cache.rs` -- the
   helper. 3-4 fns: `is_svg`, `cache_path_for`,
   `write_to_cache`, `transform_value`.
3. **Wire it in**. `mlpl-repl`'s display path
   (both local and connect mode) routes
   `Value::Str` through `viz_cache::transform_value`
   before printing.
4. **Tests**.
   `crates/mlpl-cli/tests/viz_cache_tests.rs` --
   SVG detection, cache path generation
   (deterministic from content hash), write +
   read roundtrip in a `tempfile::TempDir`.
5. **`docs/using-cli-server.md`** retrospective +
   user guide. Sections: status (Saga 21 /
   v0.17.0), what this is about, `mlpl-serve`
   quickstart (`mlpl-serve --bind ... --auth ...`,
   then `mlpl-repl --connect`), the multi-
   client picture (web UI not yet rerouted -- web
   stays WASM in MVP; that's a future saga), the
   CLI viz cache strategy + env var, the security
   posture (constant-time token compare,
   `--bind 0.0.0.0` requires auth, no proxy yet),
   non-goals deferred to follow-up sagas.
6. **`docs/configurations.md`**. CLI server
   column should now have actual values for the
   capabilities marked "yes (proxy)" today;
   refresh footnotes [3] / [7] to point at
   `using-cli-server.md` as the shipped
   reference. Note that browser unblocking via
   the proxy itself remains future work.

## Phase 4 -- release (1 step)

### Step 004 -- release v0.17.0

1. Bump `Cargo.toml` workspace.package.version
   `0.16.0 -> 0.17.0`. Minor-level bump (new
   binary crate + new client surface).
2. `CHANGELOG.md`: new v0.17.0 section above
   v0.16.0 (Saga 19's release). Document the
   `mlpl-serve` skeleton, the
   `mlpl-repl --connect` client, the CLI viz
   cache strategy, the new `docs/using-cli-
   server.md`, the deferred non-goals (proxy,
   SSE, cancellation, desktop GUI, Emacs, web UI
   re-routing).
3. `docs/saga.md`: Saga 21 retrospective above
   Saga 19 (newest first).
4. `docs/status.md`: Saga 21 row moves from
   Planned to Completed; "Next saga to start"
   pointer updates to "(dev host move to
   Linux), then Saga 17".
5. `cargo build --release`.
6. `./scripts/build-pages.sh` -- the web app
   itself doesn't change in MVP, but the
   docs/configurations.md update means the docs
   served from the live demo are stale; rebuild
   anyway to be safe and commit pages/.
7. `./scripts/gen-changes.sh` and commit
   refreshed CHANGES.md.
8. `/mw-cp` quality gates.
9. Tag `v0.17.0` locally; DO NOT push without
   explicit user confirmation (v0.13.0 / v0.14.0
   / v0.14.1 / v0.15.0 / v0.16.0 cadence).
10. `agentrail complete --done`.

## Dependency graph

```
001 mlpl-serve skeleton + sessions + eval
        |
002 mlpl-repl --connect client
        |
003 CLI viz cache + docs
        |
004 release v0.17.0
```

Sequential; each step depends on the previous.

## After Saga 21

Per `docs/status.md` the intended sequence is
**Saga 21 -> dev host move to Linux -> Saga 17
(CUDA + distributed) -> Saga 18 (distillation /
ICL/ICRL / engram memory)**. The post-MVP items
deferred above (LLM proxy, SSE, cancellation,
desktop GUI, Emacs client, web UI re-routing)
fold into a follow-up saga after Saga 21 ships
and the server contract proves stable.
