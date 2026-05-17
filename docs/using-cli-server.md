# Using MLPL with the CLI Server (`mlpl-serve`)

> **Status:** reference. Shipped in Saga 21 (v0.17.0)
> and extended in Saga 21.5 (v0.20.0). The MVP server
> contract -- sessions + eval + inspect + health +
> `mlpl-repl --connect` + the CLI viz cache -- held
> through one quarter of real use; Saga 21.5 then
> lifted the deferred items into the main body of this
> doc:
>
> - **SSE streaming eval** (`POST /v1/sessions/{id}/eval_stream`)
> - **Cancellation** (`POST /v1/sessions/{id}/cancel` + a
>   cooperative interrupt token in `eval_program_value`)
> - **Visualization storage URLs** (`POST /v1/viz` /
>   `GET /v1/viz/{sha}.{ext}`; multi-format detect for
>   SVG / HTML / PNG / JPEG / JSON)
> - **Session persistence across restarts** (JSON-on-
>   disk via `--persist <dir>`; debounced flush; load
>   on startup)
> - **Session reattach across REPL restarts** (`mlpl-repl
>   --connect <url> --session <id> --token <tok>`)
> - **Web UI in connect mode** (`apps/mlpl-web` driving
>   a remote `mlpl-serve` via the dual `Evaluator`
>   trait; CORS allow-list)
> - **f32 + u8 dtypes on the MLX peer wire** (tagged
>   3-variant union for training params + image
>   tensors)
>
> Still deferred to future sagas: server-side LLM
> proxy (security review), WebSocket surface, Emacs /
> ratatui / desktop-GUI clients. See the Non-goals
> section at the bottom for the short list.

## What this is about

Until Saga 21, MLPL had two surfaces -- the local
CLI REPL (`mlpl-repl`) and the browser-only WASM
REPL (`apps/mlpl-web`) -- with nothing shared
between them at runtime. Saga 21 adds the missing
piece: a long-running MLPL interpreter exposed as a
REST server (`crates/mlpl-serve`), with thin clients
connecting to it. One server, many clients.

Use cases the MVP enables today:

- **Attach a CLI REPL to a remote session.** Run
  `mlpl-serve` on a beefier host, then
  `mlpl-repl --connect <url>` from a laptop --
  workspace state lives on the server, the
  client is just I/O.
- **Iterate on a long-lived MLX session.** Saga
  14's MLX backend only runs natively on Apple
  Silicon; running `mlpl-serve --features mlx` on
  the AS host keeps MLX-accelerated training
  reachable from any client (CLI today; web /
  Emacs / TUI in follow-up sagas).
- **Stop dumping raw `<svg>` XML in the
  terminal.** The CLI viz cache strategy writes
  returned SVG strings to a content-addressed
  cache dir and prints `viz: <path>` instead.
  Works in both local and connect modes.

The proxy that would unblock browser
`llm_call`-against-localhost-Ollama is **post-MVP**
and lands in a follow-up saga (a careful security
review for the allow-list config + env-var
secret handling is wanted before that ships).

## What's new in Saga 21.5

Streaming, cancellation, viz storage, persistence,
reattach, and the web REPL connect mode are all
shipped. Each gets its own section below:

- [Streaming and cancellation](#streaming-and-cancellation)
  -- partial output during long `train { }` loops; a
  cooperative `/cancel` endpoint that interrupts the
  running eval.
- [Visualization storage endpoint](#visualization-storage-endpoint)
  -- `POST /v1/viz` stores SVG / HTML / PNG / JPEG /
  JSON bytes content-addressed and returns a fetchable
  URL; the connect-mode REPL and the web REPL both use
  it instead of writing the bytes to disk locally.
- [Session persistence](#session-persistence)
  -- `mlpl-serve --persist <dir>` snapshots each
  session's variables to JSON on disk and loads them
  back on startup, so a server restart doesn't lose
  workspace state.
- [Session reattach](#session-reattach) -- a REPL
  client can pass `--session <id> --token <tok>` to
  rejoin an existing server-side session instead of
  spawning a fresh one.
- [Web UI in connect mode](#web-ui-in-connect-mode) --
  the browser REPL at `apps/mlpl-web` runs against
  either its in-process WASM evaluator (default) or
  a remote `mlpl-serve` (when a `?server=<url>` query
  string is supplied).
- [MLX peer wire: f32 and u8 dtypes](#mlx-peer-wire-f32-and-u8-dtypes)
  -- the wire format grew from f64-only to a tagged
  3-variant union so image tensors and training params
  don't pay the f64 cost.

## `mlpl-serve` quickstart

```bash
# Build + run on the default loopback bind.
cargo run -p mlpl-serve --release -- \
    --bind 127.0.0.1:6464 \
    --auth required
# stderr: mlpl-serve listening on http://127.0.0.1:6464 (auth=Required)
```

Two-step "create a session, then eval a program" by
hand:

```bash
# 1. POST /v1/sessions to get a session id + token.
curl -X POST http://127.0.0.1:6464/v1/sessions
# {"session_id":"<uuid>","token":"<32-char>"}

# 2. POST /v1/sessions/<id>/eval with the bearer token.
curl -X POST http://127.0.0.1:6464/v1/sessions/<id>/eval \
    -H "Authorization: Bearer <token>" \
    -H "Content-Type: application/json" \
    -d '{"program":"iota(5) + 1"}'
# {"value":"[1 2 3 4 5]","kind":"array"}
```

`GET /v1/health` checks liveness (no auth).
`GET /v1/sessions/<id>/inspect` returns a structured
workspace snapshot (variables, models, tokenizers,
experiments) for client-side slash-command rendering.
The full endpoint catalog + JSON shapes + error
codes live at `contracts/serve-contract/sessions-and-
eval.md`.

## `mlpl-repl --connect <url>`

```bash
# Server side
mlpl-serve --bind 127.0.0.1:6464 --auth required &

# Client side (anywhere on the same network)
mlpl-repl --connect http://127.0.0.1:6464
```

The client creates a session for you on startup --
no manual `curl` dance. Each line you type POSTs to
`/eval`; the local `Environment` is unused, so the
session state lives on the server until the server
is restarted.

Slash commands route as follows:

| Command | Where it runs |
|---|---|
| `:vars`, `:models`, `:tokenizers`, `:experiments`, `:wsid` | Server (`/inspect` round-trip, rendered locally). |
| `:ask <q>` | Local Ollama. The server-side `:ask` framing path is a follow-up; today connect-mode `:ask` reads `OLLAMA_HOST` / `OLLAMA_MODEL` env vars and forwards the question without server workspace context. |
| `:help` | Client-static. Lists what's supported in connect mode. |
| Other `:`-prefixed | "(not supported)" message; `:trace`, `:describe`, `:builtins`, `:fns` etc. land in follow-up sagas. |

`--connect` is incompatible with `-f`, `--file`,
`--data-dir`, `--exp-dir` -- they all assume a local
`Environment`. Combining them errors and exits with
code 2.

The full client contract lives at
`contracts/repl-contract/connect.md`.

## CLI viz cache (`MLPL_CACHE_DIR`)

`mlpl-repl` (both local and `--connect` modes) used
to print raw `<svg>` XML inline whenever a viz
primitive returned a string. Saga 21 step 003
replaces that with a content-addressed cache:

- An SVG return value is written to
  `$MLPL_CACHE_DIR/<sha256-prefix-12chars>.svg`
  (default: `dirs::cache_dir().join("mlpl")` --
  `~/Library/Caches/mlpl/` on macOS,
  `~/.cache/mlpl/` on Linux).
- The terminal prints `viz: <full-path>` in place
  of the XML.
- Non-SVG return values pass through unchanged.

Override the cache dir per-session:

```bash
MLPL_CACHE_DIR=/tmp/mlpl-viz mlpl-repl
```

Or use the back-compat `--svg-out <dir>` flag,
which sets the cache dir for the local-mode REPL
process only:

```bash
mlpl-repl --svg-out /tmp/mlpl-viz
```

Filenames are deterministic in content -- the same
viz output written twice ends up at the same path,
so repeated calls do not accumulate junk.

Saga 21.5 grew the format table beyond SVG: see
[Visualization storage endpoint](#visualization-storage-endpoint)
for the multi-format detect + the `/v1/viz` storage
path that mints fetchable URLs instead of writing the
bytes locally.

## Streaming and cancellation

The MVP `/eval` endpoint is request/response: the
server runs the program to completion, then returns
`{value, kind}`. That's fine for short calls and bad
for long `train { }` loops where the user wants to see
loss numbers as they land.

Saga 21.5 ships `POST /v1/sessions/{id}/eval_stream`
as the streaming variant. The wire format is
Server-Sent Events; each frame is a single JSON
object on a `data:` line:

```text
data: {"kind":"metric","name":"loss","step":1,"value":0.84}
data: {"kind":"metric","name":"loss","step":2,"value":0.61}
data: {"kind":"metric","name":"loss","step":3,"value":0.49}
data: {"kind":"final","value":"[0.49]","kind_inner":"array"}
```

Frame kinds:

- `metric` -- one row per `loss_metric = ...` or
  `accuracy_metric = ...` assignment inside the
  program. Carries `name`, `step`, `value`.
- `final` -- emitted exactly once at the end. Carries
  the final `{value, kind_inner}` pair the
  request/response `/eval` endpoint would have
  returned.
- `error` -- emitted instead of `final` if the eval
  failed. Carries `{message}`.
- `cancelled` -- emitted instead of `final` if the
  eval was interrupted via `/cancel`. Carries `{step,
  partial_losses}` so the client can render the loss
  curve it managed to produce.

The CLI and web REPLs both consume `eval_stream` for
any program that contains a `train { }` block;
non-training calls keep using `/eval` to avoid the SSE
overhead.

`POST /v1/sessions/{id}/cancel` sets a cooperative
interrupt token. The next step boundary inside
`eval_program_value` checks the token and short-
circuits with `EvalError::Cancelled { step,
partial_losses }`; the streaming endpoint translates
that into a `cancelled` SSE frame. The interrupt
token is per-session, scoped to the in-flight eval
only -- a subsequent `/eval` or `/eval_stream` on the
same session starts with a fresh (un-set) token.

In the CLI REPL, Ctrl+C during a streaming eval POSTs
`/cancel` for the active session before the local
SIGINT handler tears down the client; the streamed
output is preserved up to the cancel point. The web
REPL exposes a "Cancel" button that does the same.

## Visualization storage endpoint

`POST /v1/viz` accepts a content-typed body (SVG /
HTML / PNG / JPEG / JSON) and returns a JSON object
with a content-addressed URL:

```bash
curl -X POST http://127.0.0.1:6464/v1/viz \
    -H "Authorization: Bearer <token>" \
    -H "Content-Type: image/svg+xml" \
    --data-binary @plot.svg
# {"url":"/v1/viz/<sha256-prefix-12>.svg","sha":"<sha256>",
#  "format":"svg","bytes":1834}
```

`GET /v1/viz/{sha}.{ext}` returns the bytes with the
right `Content-Type` header. The store is in-memory
on the server (cleared on restart); persistence is a
follow-up if the use case appears.

Format detect lives at `crates/mlpl-cli/src/viz_format.rs`
and is keyed off the leading bytes -- not the program
that produced them -- so it works equally well for a
hand-rolled `svg(...)` string, a `to_html(...)`
table, or a `render_image(...)` PNG buffer. The
detect table grew beyond Saga 21's SVG-only `is_svg_string`:

| Format | Source signal |
|---|---|
| `svg` | leading `<svg` or `<?xml...<svg` |
| `html` | leading `<html` or `<!DOCTYPE html>` |
| `png` | 8-byte PNG magic (`89 50 4E 47 0D 0A 1A 0A`) |
| `jpeg` | leading `FF D8 FF` |
| `json` | leading `{` or `[` (UTF-8 parse-as-JSON validates) |

In CLI connect mode, `mlpl-repl` calls `/v1/viz` to
upload returned viz bytes and prints
`viz: <server-url>` (full URL, copy-pasteable to a
browser); the local cache dir is bypassed when in
connect mode so two clients connected to the same
server see the same URL. Local-mode REPL behavior is
unchanged.

The web REPL fetches `/v1/viz/{sha}.{ext}` directly
(same-origin or via the CORS allow-list) and renders
the bytes inline -- no roundtrip through a file path.

## Session persistence

`mlpl-serve --persist <dir>` enables JSON-on-disk
persistence. Each session's slimmed-down workspace
(variables only -- models and optimizer state are
still in-memory only, deferred to a follow-up) is
flushed to `<dir>/<session-id>.json` with a debounce
window so back-to-back `eval` calls don't thrash the
disk:

```json
{
  "persist_version": 1,
  "session_id": "<uuid>",
  "token": "<32-char>",
  "created_at": 1700000000,
  "last_eval_at": 1700000123,
  "vars": [
    {"name": "x", "value_json": "[1, 2, 3]"},
    {"name": "loss_curve", "value_json": "0.49"}
  ]
}
```

On startup, `mlpl-serve --persist <dir>` rehydrates
every session file in the directory. The
`persist_version: 1` header is the upgrade hook -- a
later saga that grows the persisted set (models,
optimizer state, experiment log) bumps to v2 and a
v1 loader path can keep working.

The slim subset is deliberate: serializing a Model
tape or an Optimizer's running state would have
forced a big serialization audit across `mlpl-eval`,
and the variables-only cut covers "I want to come
back to my numbers tomorrow" without that scope.

## Session reattach

`mlpl-repl --connect <url>` defaults to creating a
new session on the server. To rejoin an existing
session (e.g., after the REPL process restarted but
the server kept running), pass the session id and
token:

```bash
mlpl-repl --connect http://127.0.0.1:6464 \
    --session 9c2e4...-... \
    --token abcd1234...
```

The client calls `GET /v1/sessions/{id}` to verify the
session exists and the token authenticates, then
prints a welcome banner showing the session id,
created-at timestamp, last-eval timestamp, and
variable count before dropping into the prompt:

```text
mlpl-repl 0.20.0 -- reattached to session 9c2e4...-... (12 vars,
                    created 2026-05-14T10:22Z, last eval 32m ago)
```

If `--session` is passed without `--token`, or with a
token that doesn't authenticate, the client errors
out with exit code 2 rather than silently spawning a
fresh session.

## Web UI in connect mode

The browser REPL at `apps/mlpl-web` runs in WASM by
default. Saga 21.5 added a remote mode driven by a
`?server=<url>` query string:

```text
# Default (in-process WASM):
https://sw-ml-study.github.io/sw-mlpl/

# Remote (against an mlpl-serve running on the LAN):
https://sw-ml-study.github.io/sw-mlpl/?server=http://my-mac.local:6464
```

The split happens behind an `Evaluator` trait
(`apps/mlpl-web/src/eval.rs`):

- `WasmEvaluator` -- the original path. Runs the
  parser + evaluator entirely in the browser via the
  WASM build of `mlpl-eval`.
- `RemoteEvaluator` -- POSTs each line to
  `/eval` or `/eval_stream` on the configured
  server. Renders SSE metric frames as live updates
  on the loss-curve chart. Fetches viz URLs via the
  CORS-enabled `/v1/viz/...` path.

`mlpl-serve --cors-allow <origin>` is the server-side
toggle that lets a non-same-origin browser hit the
API. The default deny-by-default posture matches the
MVP's "loopback + LAN-only" stance; pages-hosted
demos want an explicit allow-list rather than `*`.

The web REPL's slash commands map to the same
`/inspect` round-trips as the CLI client. Streaming
training pages the loss-curve chart line-by-line; a
"Cancel" button POSTs `/cancel`. If the WebSocket
upgrade ever ships, the trait's `start_stream` method
is the natural slot for it.

## MLX peer wire: f32 and u8 dtypes

`services/mlpl-mlx-serve` is the MLX peer service the
orchestrator talks to via `device("mlx") { ... }`.
Saga R1 shipped an f64-only wire format; Saga 21.5
extended that to a 3-variant tagged union:

| Tag | Const | Bytes/elem | Use case |
|---|---|---|---|
| `0` | `DTYPE_F64` | 8 | default; legacy compatibility |
| `1` | `DTYPE_F32` | 4 | training params, activations |
| `2` | `DTYPE_U8` | 1 | image / mask tensors |

The wire is versioned (`version: u32 = 1`) and the
tag is a single byte after the version field. The
peer's promotion ladder lifts u8 to f32 on the first
arithmetic op; f32 mixed with f64 promotes to f64.
Decode always returns an f64 `DenseArray` because
that's the orchestrator's internal representation.

`encode_tensor(arr)` (no `_as`) stays byte-for-byte
identical to R1's f64-only path. New callers that
want a smaller wire use `encode_tensor_as(arr,
DTYPE_F32)` or `encode_tensor_as(arr, DTYPE_U8)`.
The f32 path saves ~50% on training param transport;
u8 saves ~87% on image batches. See
`services/mlpl-mlx-serve/src/wire.rs` for the
encoding details and `tests/wire_dtype_tests.rs` for
the round-trip + size-delta tests.

## Security posture

- **Constant-time bearer-token compare.** Uses
  `subtle::ConstantTimeEq` so timing oracles can't
  fish out the token character-by-character.
- **Non-loopback binds require `--auth required`.**
  `--bind 0.0.0.0` (or any non-loopback address)
  with `--auth disabled` refuses to start. The
  default config (`--bind 127.0.0.1:6464` +
  `--auth required`) is the safe baseline.
- **Tokens are 32 alphanumeric chars from the
  thread-local CSPRNG.** ~190 bits, fine for
  loopback / LAN. A future saga can swap in
  `OsRng` + a longer alphabet if the threat model
  changes.
- **Sessions never expire** in MVP. Restarting
  the server is the only way to clear them.
- **No LLM proxy yet.** Browser `llm_call` is
  still blocked by CORS even when the CLI server
  is running. The proxy that lets the server
  call Ollama on the browser's behalf is a
  follow-up saga -- the security review there
  matters because the proxy needs an explicit
  allow-list of upstream LLM endpoints + env-var
  secret handling.

## Multi-client picture today

- **`mlpl-repl --connect` (CLI)** -- shipped in
  Saga 21. Saga 21.5 added `--session <id> --token
  <tok>` reattach + Ctrl-C cancel + SSE streaming
  for `train { }` blocks.
- **`apps/mlpl-web` (browser)** -- runs in WASM by
  default; `?server=<url>` switches to remote-eval
  mode via the `Evaluator` trait + CORS allow-list.
  Shipped in Saga 21.5.
- **ratatui TUI** -- future saga. Same REST
  contract as the CLI client.
- **Emacs client** -- future saga. The
  visualization storage endpoint (now shipped in
  21.5) is the dependency that was blocking it;
  Emacs can now render SVG / PNG inline by
  fetching `/v1/viz/{sha}.{ext}`.
- **Desktop GUI (tauri / wry)** -- future saga.

## Non-goals (still deferred)

The 21.5 items above closed most of the original
deferred list. What remains:

- **Server-side LLM proxy with allow-list.** Still
  unshipped. Needs the security review for
  allow-list config + env-var secret handling
  before it can land; tracked separately at
  `docs/saga.md` ("Server-side LLM proxy" row
  under planned sagas) and gated on Saga 19's
  HTTP-path consolidation, which is done.
- **WebSocket surface.** REST + SSE turned out to
  be enough for every 21.5 use case. WebSocket
  would let the server push to the client between
  evals (e.g., progress on a background training
  job that another client started); no concrete
  use case appears yet.
- **Persistence of models + optimizer state.**
  Saga 21.5's `--persist` ships variables only.
  Models and optimizer state stay in-memory; a
  later saga can grow the `persist_version`
  schema once a concrete need surfaces.
