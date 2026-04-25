# `mlpl-serve` Sessions + Eval Contract (Saga 21 step 001)

## Purpose

`mlpl-serve` exposes a long-running MLPL interpreter
as a REST API: clients create a session, get a
bearer token, and POST programs to evaluate against
the session's `Environment`. This contract pins the
MVP endpoints; the `inspect` endpoint (step 002),
LLM proxy, SSE streaming, cancellation, and
persistence are all explicit non-goals here.

The server is one binary serving many clients.
Step 002 ships `mlpl-repl --connect` as the first
real client. Web UI rerouting and other client
surfaces are deferred to follow-up sagas.

## Endpoints

### `POST /v1/sessions`

Create a new session. **No authentication
required** -- this endpoint is how a client gets
its bearer token.

Request: empty body (or any JSON, ignored).

Response (`200 OK`):

```json
{"session_id": "<uuid-v4>", "token": "<random-32-char>"}
```

The `session_id` is a UUIDv4 string. The `token` is
32 alphanumeric characters from a CSPRNG. Both are
required by every subsequent authenticated call;
losing the token means the session is unreachable
(no recovery -- create a new one).

### `POST /v1/sessions/{session_id}/eval`

Evaluate an MLPL program against the session's
`Environment`. **Authenticated** (bearer token).

Request:

```http
POST /v1/sessions/<id>/eval
Authorization: Bearer <token>
Content-Type: application/json

{"program": "iota(5) + 1"}
```

Response (`200 OK`) on success:

```json
{
  "value": "<stringified-value>",
  "kind": "array" | "string" | "model" | "tokenizer"
}
```

`value` is the `Display` formatting of the result
(arrays print like `[1 2 3 4 5]`, strings print
verbatim, models print as `<model>`, tokenizers
print as `<tokenizer: ...>`). Step 002 introduces
the structured `inspect` endpoint for richer
client rendering; the MVP eval response keeps the
simple stringified form.

Error responses:

- **`401 Unauthorized`** -- missing
  `Authorization` header, malformed `Bearer`,
  or wrong token. Body:
  `{"error": "missing or invalid authorization"}`.
- **`404 Not Found`** -- `session_id` does not
  exist. Body:
  `{"error": "unknown session"}`.
- **`400 Bad Request`** -- the program failed to
  lex, parse, or evaluate. Body:
  `{"error": "<EvalError or parse error message>"}`.
  The error message is the same one
  `mlpl-repl` would print locally.

### `GET /v1/health`

Liveness check. **No authentication required.**

Response (`200 OK`):

```json
{"status": "ok", "version": "<crate version>"}
```

The `version` field reads `CARGO_PKG_VERSION` at
compile time (currently `0.17.0` once Saga 21
step 004 lands; `0.16.0` until then since step 001
ships before the version bump).

## Security posture

- **Token compare is constant-time.** Uses
  `subtle::ConstantTimeEq` so timing oracles
  can't fish out the token character-by-character.
- **`--bind 0.0.0.0` requires `--auth required`.**
  Refusing to start otherwise is a hard
  precondition. `--bind 127.0.0.1` (the default)
  may run with `--auth disabled` for ergonomic
  loopback testing, but anything that listens
  on a non-loopback address MUST authenticate
  every eval request.
- **Tokens are 32 alphanumeric characters from
  `rand::distributions::Alphanumeric`.** Not
  cryptographic per se but enough entropy
  (~190 bits) for the loopback / LAN threat
  model. A future saga can swap in
  `rand::rngs::OsRng` + a longer alphabet if
  the threat model changes.
- **Sessions never expire** in MVP. Restarting
  the server is the only way to clear them.
  Persistence + token rotation are future-saga
  concerns.

## CLI flags

- `--bind <host:port>` (default
  `127.0.0.1:6464`). Must be a parseable
  `SocketAddr`.
- `--auth <required|disabled>` (default
  `required`). `required` enables the bearer
  middleware on `/eval` (and the future
  `/inspect`); `disabled` skips it entirely.
  `disabled` requires `--bind` to be a
  loopback address; the server prints an error
  and exits non-zero if combined with a
  non-loopback bind.

## Programmatic entry (test harness)

`mlpl-serve` ships as a binary AND a library so
integration tests can spin up the server in-
process:

```rust
use mlpl_serve::server::{build_app, run};
use mlpl_serve::auth::AuthMode;

// For tests: bind a random port, run on it.
let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
let addr = listener.local_addr().unwrap();
let app = build_app(AuthMode::Required);
tokio::spawn(async move {
    axum::serve(listener, app).await.unwrap();
});

// addr is now usable for reqwest from the same process.
```

`build_app` returns an `axum::Router` already wired
with the session map state and the auth middleware.
`run(addr, auth_mode)` does the safety check then
binds + serves; the binary `main` is a thin shell
around it.

## Module layout

- `crates/mlpl-serve/src/main.rs` -- arg parsing
  + tokio runtime + `run` orchestration. 3-4 fns,
  under the 7-fn cap.
- `crates/mlpl-serve/src/lib.rs` -- pub re-exports
  for tests.
- `crates/mlpl-serve/src/server.rs` -- `AppState`,
  `build_app`, `run`, `ServerError`. 3-5 fns.
- `crates/mlpl-serve/src/sessions.rs` --
  `Session` struct, `SessionMap` type alias,
  `new_map`, `create_session`, `generate_token`.
  4-5 fns.
- `crates/mlpl-serve/src/handlers.rs` -- one fn
  per route handler (`create_session_handler`,
  `eval_handler`, `health_handler`). 3 fns +
  small helpers as needed; under cap.
- `crates/mlpl-serve/src/auth.rs` -- `AuthMode`
  enum + `auth_middleware` middleware fn +
  `extract_bearer` helper. 2-3 fns.

## Non-goals (deferred)

These items appear in the design brief
(`docs/plan.md`) but are explicit non-goals for
step 001. They land in step 002, step 003, or
follow-up sagas:

- **`GET /v1/sessions/{id}/inspect`.** Step 002.
- **Server-side LLM proxy with allow-list.**
  Follow-up saga after the MVP server proves
  stable. Needs careful security review (allow-
  list config, env-var secrets, rate limiting)
  before shipping.
- **Server-Sent-Events streaming eval.**
  Follow-up saga.
- **Cancellation / interrupt.** The MVP `eval`
  endpoint blocks until the program finishes.
- **Persistence across restarts.** Sessions are
  in-memory only.
- **Visualization storage URLs.** SVGs returned
  by the eval endpoint are passed through as
  strings; the CLI viz cache (step 003) puts
  them on the local filesystem. Server-side
  storage + URL minting is a follow-up.
- **Web UI re-routing to call origin.** Today's
  `apps/mlpl-web` runs entirely in WASM. Pointing
  it at `mlpl-serve` instead is a non-trivial
  change worth its own scope; the MVP just
  exposes the API.
- **WebSocket surface.** Despite the saga title
  ("REST + WebSocket"), the MVP is REST-only.
  WebSocket lands once a use case (streaming,
  push notifications) needs it.
