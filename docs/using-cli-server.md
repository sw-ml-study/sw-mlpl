# Using MLPL with the CLI Server (`mlpl-serve`)

> **Status:** reference. Shipped in Saga 21 (v0.17.0).
> MVP scope: server skeleton + sessions + eval +
> inspect + health + the `mlpl-repl --connect` client +
> the CLI viz cache strategy. Server-side LLM proxy,
> SSE streaming, cancellation, persistence, and web UI
> re-routing are all explicit non-goals -- see the
> Non-goals section below.

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

Other formats (PNG, HTML, JSON) are deferred --
`is_svg_string` only detects SVG today. The format
table can grow as new viz return types ship.

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

- **`mlpl-repl --connect` (CLI)** -- shipped, this
  saga's first client.
- **`apps/mlpl-web` (browser)** -- still runs
  entirely in WASM. Pointing it at `mlpl-serve`
  instead is a non-trivial change worth its own
  scope; a follow-up saga.
- **ratatui TUI** -- future saga. Same REST
  contract as the CLI client.
- **Emacs client** -- future saga. Streaming SVG
  / PNG render in-buffer needs the visualization
  storage URL endpoint (also future) before it
  can render anything but text.
- **Desktop GUI (tauri / wry)** -- future saga.

## Non-goals (deferred)

These items appeared in the saga design brief and
were carved out as post-MVP. Each lands in a
follow-up saga after the MVP server contract proves
stable:

- **Server-side LLM proxy with allow-list.** The
  endpoint that lets browser / connect-mode
  clients call `llm_call` against a server-side
  allow-listed Ollama. Needs a security review
  for allow-list config + env-var secret handling.
- **Visualization storage URLs.** A server-side
  endpoint that mints URLs the client can fetch
  the SVG / PNG back from, replacing the
  client-local cache dir for connect-mode. The
  Emacs client and a future browser-via-server
  configuration both want this.
- **Server-Sent-Events streaming eval.** Today
  every `eval` is a single request/response.
  SSE is the natural shape for partial output
  (loss curves during a long `train { }` loop,
  streaming `llm_call`).
- **Cancellation / interrupt.** The MVP eval
  endpoint blocks until the program finishes.
  A cancel endpoint + cooperative interrupt
  point in `eval_program` is the path.
- **Persistence across restarts.** Sessions live
  in process memory only.
- **Web UI re-routing to call origin.** Today's
  `apps/mlpl-web` runs in WASM. Pointing it at
  `mlpl-serve` is a non-trivial frontend
  refactor; the MVP just exposes the API.
- **Reattach across REPL restarts.** Sessions
  persist on the server but the client generates
  a fresh one on every connect. A
  `--session <id>` flag could re-attach if the
  user has the token; not shipped today.
- **WebSocket surface.** Despite the saga title
  ("REST + WebSocket"), MVP is REST-only.
- **Other viz formats in the cache.** Today
  `is_svg_string` is the only detector. PNG /
  HTML / JSON cache paths are mechanical to add
  once a use case exists.
