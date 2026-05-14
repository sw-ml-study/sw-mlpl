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

### `GET /v1/sessions/{session_id}/inspect`

Workspace snapshot for client-side slash-command
rendering (added in Saga 21 step 002 for the
`mlpl-repl --connect` client). **Authenticated**
(bearer token).

Response (`200 OK`):

```json
{
  "vars": [
    {"name": "x", "shape": [3, 4], "is_param": false},
    {"name": "y", "shape": [], "is_param": true}
  ],
  "models": ["mdl"],
  "tokenizers": ["tok"],
  "experiments": ["sweep-v1"],
  "more": 0
}
```

- `vars` is sorted alphabetically by name and capped
  at 200 entries; the `more` field reports how many
  variables were truncated (0 when the snapshot was
  complete).
- `models`, `tokenizers`, `experiments` are sorted
  alphabetically. `experiments` is deduplicated --
  multiple runs of the same experiment name appear
  once (the per-run history lives in the
  experiment-record store, not the snapshot).
- Shapes are arrays of `usize` from
  `DenseArray::shape().dims()`. Scalars come back
  as `[]`.
- `is_param` mirrors `Environment::is_param(name)`.

Error responses match the eval endpoint:

- `401 Unauthorized` -- missing or wrong bearer.
- `404 Not Found` -- unknown session id.

### `POST /v1/sessions/{session_id}/eval_stream`

Server-Sent-Events streaming variant of `/eval`. Added in
Saga 21.5 step 001. **Authenticated** (bearer token, same as
`/eval`). Lex + parse synchronously so a malformed program
surfaces as a plain JSON `400` before the SSE stream opens;
runtime `EvalError`s land as a terminal `event: error` frame
on the stream.

Request:

```http
POST /v1/sessions/<id>/eval_stream
Authorization: Bearer <token>
Content-Type: application/json

{"program": "train 5 { loss_metric = step + 0.5 ; step + 0.5 }"}
```

Response on success: `200 OK` with
`Content-Type: text/event-stream`. The response body is an
SSE stream. Frame ordering:

1. Exactly one `event: ready` frame as the first event.
2. Zero-or-more `event: metric` frames -- one per binding
   ending in `_metric` whose value is a rank-0 scalar, fired
   after each `train { ... }` iteration. Untagged repeat /
   for blocks do not produce metric frames; only `train`'s
   per-iteration scan does.
3. Exactly one terminal frame: `event: done` on success, or
   `event: error` on any `EvalError` raised during eval.

`data:` payload schema per event kind:

```text
event: ready
data: {}

event: metric
data: {"name": "loss_metric", "step": 0, "value": 0.5}

event: done
data: {"value": "<stringified>", "kind": "array"}

event: error
data: {"error": "<EvalError display>"}
```

The `done` payload's `value` and `kind` fields match the
non-streaming `/eval` response body byte-for-byte (same
`Display` formatting, same `value_kind` mapping). Clients
that already render `/eval` output can reuse that path on
the `done` frame.

Error responses (BEFORE the SSE stream opens) match `/eval`:

- **`401 Unauthorized`** -- missing or wrong bearer. Body:
  `{"error": "missing or invalid authorization"}`.
- **`404 Not Found`** -- unknown `session_id`. Body:
  `{"error": "unknown session"}`.
- **`400 Bad Request`** -- the program failed to lex or
  parse. Body: `{"error": "<parse error message>"}`.
  Runtime errors do NOT use this code; they land as a
  terminal `event: error` frame on a `200` response.

Operational notes:

- The handler removes the session from the session map for
  the duration of the eval and re-inserts it on completion,
  so concurrent `/eval` / `/eval_stream` calls on the same
  session id receive a `404` while the prior eval is still
  running. Parallel calls on *different* sessions are
  unaffected.
- The eval runs on a `tokio::task::spawn_blocking` task so
  the async runtime stays healthy. Per-iteration metric
  emission uses `mpsc::Sender::blocking_send`, applying
  channel-bounded backpressure (capacity 64) if the client
  reads slowly.
- The MVP `/eval` endpoint is unchanged. The two endpoints
  remain peers indefinitely; `/eval` covers callers that
  don't want a streaming response.

### `POST /v1/sessions/{session_id}/cancel`

Cooperatively cancel an in-flight eval on the session.
Added in Saga 21.5 step 003. **Authenticated** (bearer
token, same as `/eval`). Idempotent: a second `POST
/cancel` after the bool is already set is a no-op.

Request:

```http
POST /v1/sessions/<id>/cancel
Authorization: Bearer <token>
Content-Type: application/json

{}
```

Body is ignored (use `{}` or omit entirely).

Response (`200 OK`) on success:

```json
{"cancelled": true}
```

Error responses match `/eval`:

- **`401 Unauthorized`** -- missing or wrong bearer.
- **`404 Not Found`** -- unknown `session_id`.

Mechanism:

- Every session has a `tokio` interrupt token kept in
  a parallel `InterruptMap` alongside the session
  record. `/cancel` reads that map (NOT the sessions
  map) so it can fire while an in-flight `/eval` or
  `/eval_stream` holds the sessions write lock or has
  removed the session for the duration of the call.
- The bearer-token check uses the copy of the token
  stored in the `InterruptMap` entry, for the same
  reason.
- Flipping the bool trips the in-flight eval at its
  next loop head (`for`, `train`, `repeat`) or
  pre-builtin checkpoint. The evaluator raises
  `EvalError::Cancelled { step, partial_losses }`;
  the SSE handler emits a terminal `event: cancelled`
  carrying both fields, and the non-streaming `/eval`
  handler returns a `400` with the Display formatting.
- Builtins do NOT yield mid-call. The cancellation
  latency floor is therefore "one op": a long-running
  `matmul` or `reduce_add` over a giant array finishes
  before the next pre-builtin check observes the
  cancel. A future saga can revisit if this bites in
  practice.
- After a cancel lands, the session is re-bindable
  for further evals. The eval handlers `reset()` the
  bool at the start of every call, so a prior cancel
  does not contaminate the next program.

SSE `event: cancelled` frame schema (Saga 21.5 step
003, in addition to the `ready` / `metric` / `done` /
`error` frames documented above):

```text
event: cancelled
data: {"step": 17, "partial_losses": [0.5, 1.5, ...]}
```

`step` is the iteration index the cancel landed on
(0 for non-loop sites); `partial_losses` is the
per-iteration loss curve accumulated by `train { ... }`
so far (empty for `for` / `repeat` / pre-builtin
sites). The session's `last_losses` binding is also
populated with the same vector, so post-cancel
`:vars` still surfaces the partial curve.

### `POST /v1/viz` and `GET /v1/viz/{id}`

Content-addressed visualization storage. Added in
Saga 21.5 step 004. **Authenticated** when
`auth_mode == Required`: the bearer must match SOME
existing session's token (auth is global, not
session-scoped, so a browser viz fetch in connect
mode does not need to know which session minted the
URL).

`POST /v1/viz`:

```http
POST /v1/viz
Authorization: Bearer <any-valid-session-token>
Content-Type: application/json

{"bytes_base64": "PHN2Zy8+", "content_type": "image/svg+xml"}
```

Response (`200 OK`):

```json
{"id": "abcdef0123456789", "url": "/v1/viz/abcdef0123456789"}
```

The `id` is the first 16 hex chars of the SHA-256
of the bytes; `url` is the path to fetch them back.
Idempotent: identical bytes always yield the same
id.

`GET /v1/viz/{id}`:

```http
GET /v1/viz/abcdef0123456789
Authorization: Bearer <any-valid-session-token>
```

Response: `200 OK` with the stored bytes as body
and the `Content-Type` header set to the value
recorded at upload time. `404 Not Found` for an
unknown id; `401 Unauthorized` for a missing or
invalid bearer.

**Eval pipeline integration.** When `eval_handler`
or the SSE `spawn_eval_task` produces a `Value::Str`
that `mlpl_cli::viz_cache::is_svg_string` identifies
as SVG, the bytes are stashed in the same store and
the eval response surfaces a `viz_url` field
pointing into it. If `MLPL_CACHE_DIR` is set in the
server's environment, the bytes are ALSO written to
`<dir>/<sha-prefix>.svg` and the response carries a
`viz_local_path` field (useful for the dev-loopback
case where `mlpl-serve` and `mlpl-repl` share a
filesystem). Both fields are omitted from the JSON
when absent.

The non-streaming `EvalResponse` shape grows to:

```json
{
  "value": "<svg>...</svg>",
  "kind": "string",
  "viz_url": "/v1/viz/abcdef0123456789",
  "viz_local_path": "/var/tmp/mlpl/abcdef012345.svg"
}
```

The SSE `done` payload mirrors the same fields when
present (still omitted by `skip_serializing_if`).

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
- `--cors-allow <origin>` (Saga 21.5 step 006,
  optional). When set, wraps the router in a
  `tower-http` `CorsLayer` that lets browsers
  on `<origin>` reach `/v1/*` with the
  `Authorization` bearer header. Required for
  the connect-mode web REPL
  (`apps/mlpl-web` running on
  `https://sw-ml-study.github.io/sw-mlpl/` or
  `http://localhost:8080`) to talk to a
  `mlpl-serve` on a different origin. Omit
  for the same-origin deploy (the
  `--static-dir <pages/>` path).

## Connect-mode web REPL (Saga 21.5 step 006)

`apps/mlpl-web` ships an `Evaluator` trait
(`src/eval.rs`) with two impls: `WasmEvaluator`
(default, runs `mlpl_wasm::WasmSession` in the
browser) and `RemoteEvaluator` (POSTs to a
remote `mlpl-serve`). The yew app picks between
them based on a new `?connect=<url>` query-string
parameter: present + non-empty -> remote;
absent or empty -> WASM.

The `Evaluator` trait is callback-based
(`fn eval(program, on_result: FnOnce(String))`)
so the WASM impl (sync, in-process) and the
REST impl (async, fetch-backed on the browser
or blocking `reqwest` on the native test
target) share one interface. Both impls return
errors as `"error: <msg>"` strings, matching
the existing `WasmSession::eval` contract so
existing call sites' red-text UI test
(`result.starts_with("error:")`) keeps working.

Server-side requirement: when the web bundle
runs on a different origin than the
`mlpl-serve` instance it points at (e.g.
GitHub Pages + a loopback dev server), the
server must be launched with
`--cors-allow <origin>` matching the bundle's
deployed origin. Otherwise the browser refuses
the fetch with a CORS error.

The trait + both impls are exercised by
`apps/mlpl-web/tests/connect_mode_tests.rs`,
which spins `mlpl-serve` up in-process on a
random loopback port and asserts that both
impls produce the same display string for
`iota(5) + 1`.

### Streaming + cancel in the web client (Saga 21.5 step 007)

`RemoteEvaluator` exposes `eval_stream(program,
on_metric, on_result)` for `train { }` programs
that emit per-iteration `_metric` frames over
SSE. The callbacks mirror the wire shapes:

- `on_metric(&RemoteMetric { name, step, value })`
  fires once per `event: metric` frame.
- `on_result(StreamOutcome)` fires exactly
  once with the terminal frame:
  - `Done { value, kind }` -- matches
    the non-streaming `/eval` response.
  - `Cancelled { step, partial_losses }` --
    when a concurrent `/v1/sessions/<id>/cancel`
    landed mid-train.
  - `Error { message }` -- HTTP failure or
    runtime `event: error`.

`RemoteEvaluator::cancel()` POSTs
`/v1/sessions/<id>/cancel` against the cached
server session. `cancel_handle()` returns a
cheap-to-clone `Send`-able handle so the native
test can fire cancel from a side thread
without making the whole `RemoteEvaluator`
`Send` (the browser is single-threaded so the
in-tick `cancel()` is the normal path).

The wire path is exercised by
`apps/mlpl-web/tests/streaming_train_tests.rs`:

- 5-iter train emits 5 metrics, terminal frame
  is `Done` with `4.5`.
- `iota(5)+1` produces the same final string
  in streaming and non-streaming modes (no
  metrics emitted).
- Cancel mid-train returns `Cancelled` with
  the partial loss curve.
- Lex / parse errors surface as `Error`.

WASM caveat: the browser path currently
buffers the SSE body to completion before
parsing (the response is read via
`gloo::net::http::Response::text().await`
rather than chunk-by-chunk). The native path
uses `BufRead::lines` and DOES stream live.
True live streaming on WASM (ReadableStream
chunk reads via web-sys) is a follow-up; the
contract above stays valid in both modes
because the final delivered frames are
identical.

The yew REPL UI wiring (cancel button, in-place
loss-curve update, tutorial lesson) is shipped
incrementally in follow-up commits; step 007
ships the wire-level Evaluator extensions +
tests, the WASM compile path, and the
pages/ rebuild.

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
  for tests, plus inline `pub mod sse` (Saga 21.5
  step 001) with `SseEvent`, `ChannelMetricSink`,
  and `eval_stream_handler`. Inline submodule
  rather than a top-level file so the crate
  module count stays at the sw-checklist budget;
  the same precedent applies to the existing
  inline `tls` mod.
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

- **Server-side LLM proxy with allow-list.**
  Follow-up saga after the MVP server proves
  stable. Needs careful security review (allow-
  list config, env-var secrets, rate limiting)
  before shipping.
- ~~**Cancellation / interrupt.**~~ Shipped in Saga
  21.5 step 003. See `POST /v1/sessions/{id}/cancel`
  above.
- **Persistence across restarts.** Sessions are
  in-memory only.
- ~~**Visualization storage URLs.**~~ Shipped in
  Saga 21.5 step 004. See `POST /v1/viz` and
  `GET /v1/viz/{id}` above.
- **Web UI re-routing to call origin.** Today's
  `apps/mlpl-web` runs entirely in WASM. Pointing
  it at `mlpl-serve` instead is a non-trivial
  change worth its own scope; the MVP just
  exposes the API.
- **WebSocket surface.** Despite the saga title
  ("REST + WebSocket"), the MVP is REST-only.
  WebSocket lands once a use case (streaming,
  push notifications) needs it.
