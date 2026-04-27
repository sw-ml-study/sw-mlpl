# MLPL deployment configurations

> Status: current through v0.18.0. Browser-only, CLI REPL,
> CLI server, and the MLX peer-service topology all have
> shipped slices. Desktop GUI and Emacs remain planned clients
> over the CLI-server contract.

MLPL is one language with multiple deployment surfaces. This
doc is the map: which configuration supports what, where the
real limitations are, and how a user picks the right one for
what they want to do.

## Quick configuration matrix

| Capability                           | Browser-only | CLI REPL | CLI server | Desktop GUI |
|--------------------------------------|:------------:|:--------:|:----------:|:-----------:|
| Interpreter (`iota`, matmul, etc.)   | yes          | yes      | yes        | yes         |
| Model DSL + autograd + training      | yes          | yes      | yes        | yes         |
| MLX backend (Apple Silicon)          | no           | yes      | yes        | yes         |
| Remote MLX peer (`mlpl-mlx-serve`) [9] | no         | via server | yes      | yes         |
| LoRA fine-tune, CPU [4]              | slow [5]     | yes      | yes        | yes         |
| LoRA fine-tune, MLX-accelerated [4]  | no [5]       | yes      | yes        | yes         |
| `estimate_train` / `estimate_hypothetical` / `feasible` [6] | yes | yes | yes | yes |
| `calibrate_device` [6]               | unreliable   | yes      | yes        | yes         |
| `llm_call(url, prompt, model)` [7]   | no [3]       | yes      | yes (proxy)| yes (proxy) |
| Inline SVG visualization             | yes          | file [1] | yes        | yes         |
| Trace export                         | no           | yes      | yes        | yes         |
| Filesystem `load("rel.csv")`         | no [2]       | yes      | yes        | yes         |
| `load_preloaded(...)` compiled data  | yes          | yes      | yes        | yes         |
| BPE tokenizer                        | yes          | yes      | yes        | yes         |
| `experiment "name" { }` -> run.json  | mem only     | disk     | disk       | disk        |
| Host process spawning                | no           | direct   | direct     | via server  |
| Call local Ollama via `:ask`         | no [3]       | yes      | yes (proxy [8])| yes (proxy [8]) |
| Call LAN Ollama                      | no [3]       | yes      | yes (proxy [8])| yes (proxy [8]) |
| Call cloud LLM (OpenAI, etc.)        | no [3]       | yes      | yes (proxy [8])| yes (proxy [8]) |
| Multi-client concurrency             | n/a          | n/a      | yes [8]    | yes [8]     |
| Session persistence across clients   | no           | no       | yes [8]    | yes [8]     |
| `mlpl-repl --connect <url>` [8]      | no           | yes      | yes        | yes         |
| CLI viz cache (`MLPL_CACHE_DIR`) [8] | n/a          | yes      | yes        | yes         |
| Attach from Emacs                    | no           | partial  | yes [8]    | yes [8]     |

[1] In CLI the visualization primitives (`svg(...)`,
`heatmap(...)`, `loss_curve(...)`, ...) write the result to a
local file path (see "CLI visualization strategy" below) rather
than displaying inline. The path is printed; the download is
separate.

[2] The browser sandbox cannot read local files. Users can
paste small inline datasets or use `load_preloaded(name)` for
compiled-in corpora. A future "proxied file read" via the CLI
server is out of scope for the browser-only config.

[3] CORS: a browser's `fetch("http://localhost:11434")` is
blocked by Ollama's default CORS policy. The CLI server
MVP shipped in Saga 21 (v0.17.0) -- see note [8] -- but
the specific server-side LLM reverse proxy that would unblock
the browser is a deferred follow-up saga (the proxy needs a
careful security review around allow-list config + env-var
secret handling before it ships). Until then, browser
`llm_call` / `:ask` remain unreachable.

[4] Saga 15 (v0.13.0). The LoRA language surface
(`freeze`, `lora`, `LinearLora`) is pure Rust and lands in
every environment the evaluator runs in. The CPU path works
in both the CLI REPL and the browser WASM. The MLX path is
native-only -- `mlpl-mlx-rt` links against Accelerate / Metal,
which WASM cannot reach. The tutorial lesson in the web REPL
runs a deliberately tiny variant (V=8, d=4, rank=2) so the
full walkthrough stays interactive; `demos/lora_finetune.mlpl`
(CPU, any host) and `demos/lora_finetune_mlx.mlpl` (Apple
Silicon CLI only) are the full-scale artifacts. The
CLI-server configuration plus the Saga R1 MLX peer service is
the path for "browser UI + MLX-accelerated training": the web
client posts programs to an orchestrator, and the orchestrator
forwards `device("mlx") { ... }` blocks to a native peer.

[5] WASM has no Metal / Accelerate path and cannot compile
`mlx-rs`, so `device("mlx") { ... }` in the browser is a
silent no-op (the one-time warning prints once per session).
At V=280 / d=32 the full-scale CPU LoRA training loop runs in
the browser but is too slow to be interactive; use the CLI
for full-scale fine-tunes, or use a CLI-server orchestrator
with a Saga R1 MLX peer.

[6] Saga 22 (v0.15.0). `estimate_train`,
`estimate_hypothetical`, and `feasible` are pure math
over the `ModelSpec` tree + lookup tables, so they
run identically in every environment -- browser
included. `calibrate_device` runs a square-matmul
benchmark on the active device; in WASM the browser
timer resolution + tab-event-loop jitter make the
measurement unreliable, so the cached default (50
GFLOPS) is preferable there. CLI users should run
`calibrate_device()` once per session (or per device
switch) to get honest wall-clock estimates.

[7] Saga 19 (v0.16.0). `llm_call(url, prompt, model)`
is the language-level builtin form of the REPL's
`:ask` command -- one POST to an Ollama-compatible
`/api/generate` endpoint, returns the completion as
a string. The browser sandbox cannot reach a
localhost Ollama server (same CORS story as note
[3]); the CLI server MVP shipped in Saga 21
(see note [8]) but the specific reverse-proxy
endpoint that would unblock the browser remains
deferred to a follow-up. Streaming SSE,
OpenAI-style tool calling, multi-turn chat
threading, request batching, and in-source auth
secrets are all explicit non-goals -- see
`contracts/eval-contract/llm-call.md` and
`docs/using-llm-tool.md` for the full deferred
list.

[8] Saga 21 (v0.17.0). The CLI server
(`crates/mlpl-serve`) MVP skeleton ships with
`POST /v1/sessions`, `POST /v1/sessions/{id}/eval`,
`GET /v1/sessions/{id}/inspect`, and
`GET /v1/health`, plus the
`mlpl-repl --connect <url>` client and the CLI
viz cache strategy (`MLPL_CACHE_DIR` /
`dirs::cache_dir()/mlpl/`, content-addressed SHA-
prefix paths replacing raw `<svg>` XML in the
terminal). The browser-unblocking LLM proxy,
visualization storage URLs, Server-Sent-Events
streaming, cancellation, persistence across
restarts, web UI re-routing to call origin, the
ratatui TUI client, the Emacs client, and the
desktop GUI wrapper are all explicit follow-up
sagas after the MVP server contract proves
stable. Cells flagged "(proxy [8])" depend on
the LLM-proxy follow-up specifically, not on
the MVP that just shipped. See
`docs/using-cli-server.md` for the full picture
plus the security posture (constant-time token
compare; non-loopback bind requires
`--auth required`).

[9] Saga R1 (v0.18.0). MLX can now run as a
separate peer service in `services/mlpl-mlx-serve/`.
The orchestrator registers it with
`--peer mlx=<url>` and forwards whole
`device("mlx") { ... }` blocks. Results stay on
the peer as opaque `DeviceTensor` handles until
the program explicitly fetches them with
`to_device("cpu", x)`. The in-process MLX feature
remains the fallback when no peer is registered.
Loopback peer URLs are the default safe shape;
non-loopback peers require `--insecure-peers`
while the longer-term peer trust model remains
future work. See `docs/using-mlx-service.md`.

## Configuration 1: Browser-only

> Today's live demo at
> <https://sw-ml-study.github.io/sw-mlpl/>. Shipped in v0.11.0
> after Saga 14.

Runs entirely in WebAssembly on the user's device. No
install, no account, no server. The interpreter, the autograd
tape, the Model DSL, the visualizations -- every MLPL feature
that does not touch the filesystem, a socket, or a GPU works
the same as in the CLI.

### What works

- Every MLPL language feature through Saga 14 except the ones
  marked "no" in the matrix above.
- Every tutorial lesson, every demo in the dropdown.
- The full Model DSL + autograd + Adam + `train { }` loop.
- Inline SVG visualization (the string output of `svg(...)`
  renders inside the REPL's scrollable history).
- `:describe` over variables, models, tokenizers, strings (for
  demo narration via `_demo`), and builtins.

### What does not work

- **MLX backend.** wasm32 cannot link `mlx-rs`. `device("mlx")
  { body }` inside the browser prints a one-time "falling back
  to CPU" warning; correctness is preserved, speed is the CPU
  baseline.
- **Filesystem reads.** `load("path.csv")` errors cleanly
  (sandbox root unset). Use `load_preloaded(name)` for
  compiled-in corpora.
- **Ollama / `:ask`.** The browser's `fetch` to
  `http://localhost:11434` is blocked by Ollama's CORS. Even if
  the user sets `OLLAMA_ORIGINS="*"`, the web REPL deliberately
  does not call out -- users navigating to the demo should not
  have browser tabs silently talking to local daemons.
- **Long training runs.** Single-threaded wasm32 on the
  browser main thread. A training loop longer than the
  browser's "unresponsive tab" timeout (~15-30 s) triggers a
  kill-or-wait dialog. Mitigations: demos pinned to tutorial-
  budget configs (V=260, d=16, block=8, 30 steps); Web Worker
  plan in `docs/worker-threads.md` moves eval off the main
  thread but is not yet shipped.
- **Trace JSON export.** No filesystem to write to. The trace
  is computed but cannot be downloaded from the browser today
  (a "download trace" button is a small follow-up item).
- **Host process spawning.** By design; the browser cannot
  spawn processes. The CLI server is where this lives.

### When to pick this configuration

- Demos, conference talks, documentation.
- First-time exploration: "does the language make sense?"
- Tutorial lessons for self-study.
- Classroom settings where installing Rust is a non-starter.

### Known limitations made clear

Every browser-only limitation from the matrix above is
surfaced in the UI when the user hits it -- `load` errors with
"filesystem access disabled"; `device("mlx")` prints the
fallback warning; the README "Links" section and the
docs/using-*.md docs are explicit that backends requiring
native dependencies are CLI-only. New limitations found in
testing should get the same treatment: error early, link the
doc, do not silently degrade.

## Configuration 2: CLI REPL only

> Today's `mlpl-repl` binary. Installed via
> `cargo install --path apps/mlpl-repl`. Shipped in v0.11.0.

Runs natively on the user's machine. The interpreter has
filesystem access, can spawn processes (e.g. for the `:ask`
HTTP call to Ollama), and -- on Apple Silicon with `--features
mlx` -- dispatches through the MLX backend.

### What works beyond browser-only

- MLX backend on Apple Silicon (`cargo run -p mlpl-repl
  --features mlx`).
- `load("rel.csv")` under a `--data-dir` sandbox root.
- `experiment "name" { }` writes `run.json` to the
  `--exp-dir`. `:experiments` shows every recorded run.
- Trace export to disk (`:trace json <path>`).
- `:ask <question>` calls a local Ollama server over plain
  HTTP (no CORS issues; the REPL is a normal HTTP client).

### What does not work

- **Inline visualization.** The terminal is not a raster
  canvas. Today `svg(...)` prints the raw SVG source. See "CLI
  visualization strategy" below for the correct answer.
- **Multi-client sessions.** One REPL = one process = one
  `Environment`. No way to attach a second window to the same
  session.
- **Web UI surface.** Users who want the inline plots, the
  demo dropdown, and the tutorial lessons must either (a) open
  the browser-only live demo separately or (b) run the CLI
  server.
- **TUI polish.** Optional future work: `ratatui`-based TUI
  with a variable panel, a loss curve widget, a status bar
  showing `:wsid` output. Not a hard dependency; the plain
  line-based REPL remains the baseline.

### CLI visualization strategy

Every visualization primitive in MLPL (`svg(...)`,
`loss_curve(...)`, `scatter(...)`, `heatmap(...)`,
`confusion_matrix(...)`, `boundary_2d(...)`) returns a string
of SVG XML that the browser renders inline. The CLI cannot
render SVG in a terminal. Three options, in order of
preference:

1. **Write to a file + print the path (default).** The REPL
   already has a `--svg-out <dir>` flag that, when set,
   captures every top-level result whose output starts with
   `<svg` and writes it as a timestamped `.svg` file. Extend
   this to be the default when SVG output is detected and the
   `--svg-out` flag is unset: write to
   `$XDG_CACHE_HOME/mlpl/svg/` (or `./svg-out/` if unset) and
   print the path. The user sees
   `[svg written to ~/.cache/mlpl/svg/2026-04-22T10-12-03.svg]`
   instead of hundreds of lines of `<svg>...<path d="...">`
   noise.

2. **Optional `$BROWSER` / `xdg-open` hand-off.** A
   `:svg-open` toggle (off by default) passes the written file
   path to the user's default browser or image viewer. Not a
   core feature -- the user is already in a terminal; if they
   wanted a browser they would have opened one -- but useful
   in workflows where the CLI is driving a presentation.

3. **A wrapping builtin that saves + returns the path.** A
   new `save_svg(payload, path)` helper that writes the SVG
   string to the given path and returns the path as a string.
   Composes with everything else: `p = save_svg(loss_curve(
   last_losses), "loss.svg")` then `:!open $p` if the shell
   supports it. Keeps the implicit auto-write simple for the
   common case and the explicit save composable for scripts.

The Emacs client (see below) is a third path: it is a CLI
client by default but can render the SVG in-buffer without
being a web browser, using Emacs' native image-mode.

### When to pick this configuration

- Development: quick interactive exploration, training a tiny
  model on a local dataset, running a `.mlpl` script with
  `mlpl-repl -f`.
- Scripting and CI: `mlpl-repl -f path.mlpl` in a pipeline.
- MLX performance work: Apple Silicon only path.
- Offline use: no network needed unless `:ask` is invoked.

## Configuration 3: CLI server (planned)

> Design target for a future saga. Not yet shipped. This is
> the "most capable" configuration: a long-running MLPL server
> that provides a REST API, serves the web UI from an origin
> it controls (so the browser CAN call it without CORS
> gymnastics), reverse-proxies to local or remote LLM
> services, and accepts CLI / TUI / emacs clients over the
> same protocol.

The problem this solves: the browser-only config is useful
for demos but walled off from host resources; the CLI-REPL
config has host access but no inline visualization, no
multi-client sessions, and no way to coordinate between
machines. The CLI server splits those roles cleanly -- the
heavy interpreter work runs on a trusted host; any number of
thin clients (web, CLI, emacs, desktop GUI) talk to it.

### Shape

```
+------------------+        HTTP(S)         +-------------------------+
| web client       | <--------------------> |                         |
| (same assets as  |                        |                         |
|  apps/mlpl-web/, |                        |                         |
|  wired to call   |      WebSocket         |      mlpl-serve         |
|  origin not      | <--------------------> |      (new crate)        |
|  same-page)      |                        |                         |
+------------------+                        |  - REST: /eval,         |
                                            |    /describe, /vars,    |
+------------------+        HTTP(S)         |    /experiments, ...    |
| CLI client       | <--------------------> |  - WebSocket: /session  |
| (mlpl-repl       |                        |    for streaming        |
|  --connect ...)  |                        |  - Static: /app         |
+------------------+                        |  - Proxy: /proxy/ollama |
                                            |    /proxy/cloud-llm     |
+------------------+        HTTP(S)         |  - Sessions: isolated   |
| emacs client     | <--------------------> |    per-token, each      |
| (image-mode)     |                        |    holds an Environment |
+------------------+                        |                         |
                                            +-----+---------+---------+
                                                  |         |
                                    reverse proxy |         | reverse proxy
                                                  v         v
                                    +--------------+  +---------------+
                                    | local Ollama |  | cloud LLM     |
                                    | 11434        |  | (OpenAI, etc.)|
                                    +--------------+  +---------------+
```

### API surface

All endpoints accept a bearer token in the `Authorization`
header. Default bind is `127.0.0.1:9001`; `--bind 0.0.0.0`
requires `--auth required` and prompts for a token on first
launch (or reads `MLPL_SERVE_TOKEN` from env).

**Sessions**

- `POST /v1/sessions` -- create a session, returns `{id,
  token}`. The token scopes every subsequent call.
- `DELETE /v1/sessions/:id` -- tear down, frees the
  `Environment`.
- `GET /v1/sessions/:id/vars` -- current `:vars` output as
  JSON (name, shape, labels, optional preview).
- `GET /v1/sessions/:id/describe/:name` -- `:describe <name>`
  output as JSON.
- `GET /v1/sessions/:id/experiments` -- `:experiments`
  registry.

**Eval**

- `POST /v1/sessions/:id/eval` -- body: `{source: "..."}`.
  Returns `{output, is_error, events, viz_urls}`. Streaming
  variant via `POST /v1/sessions/:id/eval/stream` sending
  Server-Sent Events per step of a `train { }` loop.
- `POST /v1/sessions/:id/cancel` -- cancel a running eval via
  the cooperative cancellation flag that `docs/worker-threads.md`
  already sketches.

**Visualization**

- `POST /v1/viz` -- write an SVG payload to server-side
  storage; returns a stable URL. The web client embeds this
  URL in an `<img>`; the CLI client renders `[svg: <url>]` and
  a `save_svg(url, "./foo.svg")` sugar pulls it. The emacs
  client fetches the URL and renders inline.
- `GET /v1/viz/:hash.svg` -- retrieve. 24h TTL; the
  invariant is "never leave a viz URL in a committed
  notebook."

**Proxy**

- `POST /v1/proxy/ollama/api/chat` -- forwards to
  `$OLLAMA_HOST` (default `http://localhost:11434`). Adds
  `Content-Type`, preserves streaming. No transformation of
  request or response bodies. This is the single endpoint the
  web client calls for `:ask` -- the browser has CORS with
  our origin because we set the headers; our origin has no
  CORS with Ollama because we are server-to-server.
- `POST /v1/proxy/cloud-llm/:provider` -- same shape for
  OpenAI / Anthropic / Groq / etc. The server holds the API
  keys (env vars), not the client. Mandatory allow-list per
  provider so a compromised client cannot invent an endpoint.

### Client protocols

- **Web client** -- same Yew assets as the current
  `apps/mlpl-web` but (a) the eval call goes to the origin's
  `/v1/.../eval` instead of calling `WasmSession::eval`
  directly, and (b) the `:ask` path works because the browser
  calls our origin, which holds the Ollama connection
  server-side.
- **CLI client** -- `mlpl-repl --connect <host:port>` (new
  flag). Under the hood, every line the user types becomes a
  `POST /eval`. Output renders exactly as local REPL today;
  SVG payloads are stripped to `[svg: <url>]` with the same
  auto-save behaviour as the local CLI.
- **TUI client (optional)** -- `mlpl-tui --connect ...` --
  ratatui-based, variable panel, loss curve widget, command
  history scrollback, progress bars for `train { }`. Not
  required for server correctness; additive on top of the
  CLI.
- **Emacs client** -- a new `mlpl-mode.el` that uses `url.el`
  to hit the same REST endpoints. The key win: when the
  server returns a viz URL, Emacs fetches it and uses
  `create-image` + `insert-image` to render the SVG in the
  REPL buffer. No browser involved; no external window.
- **Desktop GUI** -- a `wry` or `tauri` wrapper that starts
  `mlpl-serve` on a random localhost port and opens the web
  client inside a platform WebView. Same backend, same
  protocol; the only difference is the window chrome.

### Why this is the right shape

- **One backend.** The server is the source of truth. Every
  client is a view; adding a new client (mobile app, VS Code
  extension) is a REST integration, not a re-implementation.
- **Proxy solves CORS once.** Configure `OLLAMA_ORIGINS` once
  on the server; never ask a demo user to configure their
  Ollama.
- **Session isolation gives safe multi-tenancy.** A single
  `mlpl-serve` on a home LAN can serve multiple users
  (classroom, small team) without their state colliding.
- **No new language features required.** Everything the
  server dispatches is a language feature that already ships
  in the CLI REPL; the server is plumbing, not language.
- **Emacs is cheap.** Because the protocol is HTTP+JSON, the
  Emacs client is ~200 lines of elisp. The hard part
  (running MLPL, getting an SVG) is done server-side.

### Security posture

Running an MLPL server means exposing an arbitrary-code-
execution endpoint. The server gets these defaults:

- Binds `127.0.0.1:9001` by default; `--bind` to anything
  else requires `--auth required`.
- No token = no session. Tokens are per-session (create
  session returns one) and not reused across sessions.
- Proxy allow-list is explicit: only the providers named in
  the server config file (or via `--allow-proxy
  <provider>`) are reachable.
- Request bodies are size-capped (default 1 MB); uploads and
  proxy streams have a separate quota.
- `--read-only` flag for demo deployments: eval is allowed;
  `load` / `experiment -> disk` / `:trace json <path>` are
  disabled.
- `docs/security.md` (new doc, also planned) covers the
  threat model in full. This doc only lists defaults.

### When to pick this configuration

- **Group / team use.** Multiple people using one machine's
  compute; each gets a session.
- **"Serious" LLM integration.** `:ask` that actually works
  from the browser, because it goes through the origin's
  proxy.
- **Remote compute.** Desktop with a GPU, laptop driving it
  over the LAN.
- **Emacs workflows.** The configuration that makes "run
  MLPL from inside my editor, render plots inline, ask an LLM
  a question about my session, all in the same buffer" work.
- **Live tutorials + classroom.** Students open a web URL;
  each gets an isolated session; the instructor pushes code
  through the same REST endpoints.

## Configuration 4: Desktop GUI (future)

> A small wrapper, not a new architecture. Likely shipped
> after the CLI server ships.

Packages `mlpl-serve` and the web client assets into a single
native binary using `tauri` or `wry`. On launch:

1. Spawn `mlpl-serve` on a random localhost port with a
   generated token.
2. Open a native window hosting the web client pointed at
   `127.0.0.1:<port>` with the token pre-populated.
3. On window close, tear down the server.

User value: no terminal, no "first install Rust, then run
`cargo install`". The first-run UX matches what users
expect from a GUI app (double-click, use it). Everything
beyond the window chrome is the CLI server's responsibility.

Trade-off: we ship platform binaries (macOS .app, Windows
.exe, Linux AppImage). That is a distribution chore the CLI
server does not have. Defer until the server is stable.

## Emacs client (cross-cutting)

The Emacs client is not a configuration on its own -- it is a
client that attaches to either the CLI REPL (when no server is
running, over a persistent subprocess) or the CLI server (over
HTTP). Its distinguishing feature: in-buffer image rendering
without being a browser.

### Minimum viable flow

1. User opens a `.mlpl` file or runs `M-x mlpl-repl`.
2. Emacs spawns `mlpl-repl --emacs` (a mode flag that
   promises stable newline-terminated output, strips ANSI,
   and writes SVG payloads to a tempfile + prints a sentinel
   `[mlpl:svg <path>]`).
3. `mlpl-mode.el` reads output, matches the sentinel, and
   calls `(insert-image (create-image path 'svg nil))` to
   render inline.
4. `C-c C-a` binds to `:ask <prompt-region-or-selection>`.
   Output streams into a separate buffer.

When the CLI server is running:

5. `M-x mlpl-connect <host:port> <token>` switches
   backends. Same keybinds; REST calls replace the
   subprocess.

### Why Emacs is interesting

Emacs is the only editor that renders SVG natively in a
buffer without embedding a browser. Jupyter renders via
browser; VS Code's cell output is a webview; terminal
multiplexers do not render images at all. The Emacs client
is therefore the clearest non-browser visualization story
MLPL has.

## Cross-cutting concerns

### Visualization across configurations

Same mental model, different surface:

| Configuration | `svg(...)` in REPL produces                                 |
|---------------|-------------------------------------------------------------|
| Browser-only  | inline `<svg>` rendering, download button                  |
| CLI REPL      | `[svg written to ~/.cache/mlpl/svg/<ts>.svg]`              |
| CLI server -> | server-cached URL; web UI embeds; CLI prints `[svg: url]`  |
|   web client  |                                                             |
| CLI server -> | server-cached URL; client auto-downloads to local path     |
|   CLI client  |                                                             |
| CLI server -> | server-cached URL; Emacs fetches + `insert-image`          |
|   emacs       |                                                             |
| Desktop GUI   | same as CLI server -> web client                            |

The language semantics do not change: `svg(...)` always
returns an SVG string and always produces a value the REPL
can bind. What changes is what the surface does with that
string.

### `load_preloaded` vs `load`

- `load_preloaded(name)` works everywhere -- the corpus is
  compiled into the binary.
- `load(path)` works only where a filesystem is reachable:
  CLI REPL (under `--data-dir`), CLI server (under the
  server's configured data root), desktop GUI (under a
  user-chosen data directory at first launch).
- Browser-only has no `load` surface. Data has to be pasted,
  preloaded, or fetched via a future CLI-server proxy
  endpoint for remote files.

### Ollama / cloud LLM reachability

| Configuration | `:ask` calls reach                                     |
|---------------|-------------------------------------------------------|
| Browser-only  | nothing (`:ask` disabled in this build)               |
| CLI REPL      | `$OLLAMA_HOST` directly from the process              |
| CLI server    | `$OLLAMA_HOST` via server-side proxy                  |
| Desktop GUI   | `$OLLAMA_HOST` via the embedded server's proxy        |
| Emacs client  | whatever the attached backend reaches                 |

Cloud LLMs (OpenAI, Anthropic, ...) are unreachable from the
browser-only config by design -- the API keys would have to
live client-side, which means exposed in page source. The CLI
server holds the keys and proxies the call; the client only
sees a `/v1/proxy/cloud-llm/openai` endpoint that gates the
request through an allow-list.

### Filesystem and experiment records

- Browser-only: `experiment "name" { }` records go to
  `env.experiment_log` in memory. The tab closes -> the
  records are gone.
- CLI REPL: `--exp-dir <path>` enables on-disk
  `<path>/<name>/<timestamp>/run.json` writes.
- CLI server: server-level `--exp-dir` applies to every
  session; per-session isolation prevents two users from
  writing to each other's records.

## Migration path

v0.11.0 ships the browser-only and CLI-REPL-only
configurations. The CLI server is additive -- none of the
existing configurations change.

### Minimum viable CLI server (MVP)

A single PR / saga step lands:

1. New crate `crates/mlpl-serve` -- binary + thin HTTP
   scaffolding (axum or hyper, no tokio gymnastics -- the
   workload is IO-bound and low concurrency).
2. One endpoint: `POST /v1/sessions` + `POST
   /v1/sessions/:id/eval`. Sessions hold an `Environment`
   behind an `Arc<Mutex<...>>`; each eval is a serial call.
3. `mlpl-repl --connect <host:port>` passes input to
   `/v1/.../eval`, prints the `output` field of the reply.

After MVP, incremental steps add: proxy, visualization
storage, cancel, session GC, bearer-token auth,
`--read-only`, static asset serving for the web client, and
so on. Each step keeps the REST shape stable so the CLI
client never breaks.

### Why start with CLI server MVP instead of desktop GUI

The desktop GUI is a wrapper. It has nothing to wrap until
`mlpl-serve` exists. Start with the server; the window
chrome is last.

## Implementation status

| Component                             | Status                                              |
|---------------------------------------|-----------------------------------------------------|
| Browser-only live demo                | shipped v0.11.0                                     |
| CLI REPL + `--features mlx`           | shipped v0.11.0                                     |
| `:ask <question>` (CLI only)          | shipped v0.11.0 (Saga 19 preview)                   |
| CLI server (`mlpl-serve`)             | planned -- new saga, MVP scope above                |
| CLI client (`mlpl-repl --connect`)    | planned -- same saga                                |
| Proxy endpoints                       | planned -- same saga, after MVP                     |
| Web client wired to origin            | planned -- after server serves static assets        |
| Emacs client (`mlpl-mode.el`)         | planned -- separate small saga after server MVP     |
| TUI client                            | optional follow-up, after server MVP                |
| Desktop GUI wrapper                   | planned -- separate saga, after server is stable    |
| `:ask` in the web client (via proxy)  | lands with the server, not before                   |
| Web Worker for long runs              | planned -- `docs/worker-threads.md`, independent    |

## Related

- `docs/architecture.md` -- crate-level architecture
  (language, runtime, viz).
- `docs/repl-guide.md` -- today's REPL surface per command.
- `docs/using-mlx.md` -- the MLX backend, CLI-only.
- `docs/using-ollama.md` -- the `:ask` command, CLI-only.
- `docs/using-cuda.md` -- planned CUDA backend, CLI-only.
- `docs/worker-threads.md` -- browser responsiveness, pair
  with browser-only config.
- `docs/benchmarks.md` -- what "fast" means per
  configuration.
- `docs/mlpl-for-neural-thickets.md` -- a future demo that
  will run easily on CLI server (variant sweeps need
  multi-process fan-out).
