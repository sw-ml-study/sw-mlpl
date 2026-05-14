# Multi-client UI Follow-up Milestone (Saga 21.5)

## Why this exists

Saga 21 shipped the MVP CLI server (`mlpl-serve`) and its first
client (`mlpl-repl --connect`). The MVP carved out a deliberately
long list of follow-ups to keep the initial server contract small;
those follow-ups are documented in
`docs/using-cli-server.md#non-goals-deferred`. R1 (v0.18.0) layered
the MLX peer on top and proved the orchestrator/peer shape works.
After two saga cycles of real use, the MVP contract has held -- it
is time to pick up the deferred work.

Saga 21.5 turns the MVP server into the surface other clients
(browser, ratatui, Emacs, desktop GUI) and other workloads (long
training runs, image batches) can actually depend on. The headline
unlock is browser-runs-against-server: today `apps/mlpl-web` runs
entirely in WASM and cannot reach `mlpl-serve` at all, which
forecloses any "browser triggers training on the MLX peer" demo.

This is also the prerequisite for Saga 29 (Vision Transformer
track); the "thorough" ViT demo wants to train on the MLX peer and
have its loss curve stream into the browser REPL. Without 21.5,
that story stays hypothetical.

Goal ranking applied:

- **Utility** leads: every follow-up listed here unblocks a
  user-visible scenario the MVP could not.
- **Correctness**: the contract changes here are additive but
  load-bearing for every future client. Wire format and error
  taxonomy decisions cost more to undo later than now.
- **Educational**: live loss curves and attention-during-training
  visualization are obvious teaching wins for the web tutorial.
- **Performance** is explicitly last; streaming and dtype changes
  are about *what is possible*, not about microseconds.

## Non-goals

- **Multi-tenant isolation / capability scoping.** Sessions are
  still per-server-process; per-client capability tokens (read-only
  vs eval, scoped namespaces) are out.
- **Distributed eval / horizontal sharding.** That story belongs to
  R3.
- **Persistent storage of model weights.** Save / load model
  artifacts is still the Saga 15 follow-up. Sessions persist enough
  to survive a server-restart and rebind clients; the weights
  themselves go nowhere unless `save_model` ships separately.
- **Authentication beyond bearer tokens.** mTLS, SSO, OIDC are all
  follow-ups, gated on a real threat-model write-up.
- **WebSocket transport.** SSE covers streaming for MVP needs; the
  WebSocket variant explicitly carved out by Saga 21 stays out of
  scope here too. Revisit if a use case forces it.
- **Web client architectural rewrite.** Re-route the existing
  `apps/mlpl-web` WASM client to the server *behind a feature flag*;
  do not collapse the WASM-only path. Two modes shipped in parallel
  is fine for one saga -- the WASM mode keeps the offline tutorial
  story alive.

## Quality requirements (every step)

Identical to Saga 23. TDD, all four `cargo` gates +
`markdown-checker` + `sw-checklist` green, `/mw-cp` checkpoint,
push after every commit. Web UI changes rebuild `pages/` via
`scripts/build-pages.sh`. `.agentrail/` committed.

Server changes have an additional requirement: the
`contracts/serve-contract/` prose specs are updated *in the same
commit* as the implementation. The MVP set a precedent that the
contract is authoritative; we honor it.

## What already exists

- `crates/mlpl-serve` REST skeleton: sessions, eval, inspect,
  health, constant-time bearer auth (Saga 21).
- `mlpl-repl --connect <url>` thin client (Saga 21).
- `services/mlpl-mlx-serve` MLX peer with opaque `DeviceTensor`
  handles, strict cross-device CPU faults, `to_device("cpu", x)`
  materialization, f64-only wire (R1).
- `MLPL_CACHE_DIR` CLI viz cache (Saga 21).
- `apps/mlpl-web` WASM REPL with tutorial, demo dropdown,
  numeric-summary accordion (Saga 8 onward).

## Phases

The deferred Saga-21 follow-ups split cleanly along blast radius.
Ship the small, additive ones first; save the surface-changing
ones for the back half.

### Phase 1 -- Server-Sent-Events streaming eval (2 steps)

#### Step 001 -- SSE endpoint scaffold
New `POST /v1/sessions/<id>/eval_stream` that returns
`text/event-stream`. Server emits one `event: ready` on accept,
zero-or-more `event: metric` events for `_metric =` captures inside
the program, then `event: done` with the final value or `event:
error` with the structured error. MVP `/eval` keeps working
verbatim. Tests: SSE round-trip on a 10-step `train { }`, metric
events arrive in order, `done` payload matches the non-streaming
endpoint's response body byte-for-byte. Browser-fetch and
`reqwest::EventStream` both exercised.

#### Step 002 -- live `last_losses` over SSE in the connect client
`mlpl-repl --connect` learns a `--stream` flag (and reads
`MLPL_REPL_STREAM=1`) that routes every eval through the SSE
endpoint. The default is non-streaming so existing scripts stay
quiet. When streaming, `_metric` events redraw the same one-line
loss display the local REPL already uses for `train { }`. Tests:
the same `demos/tiny_lm.mlpl` program produces the same final
value over streaming and non-streaming paths.

### Phase 2 -- Cancellation (1 step)

#### Step 003 -- cancellation endpoint + cooperative interrupt
`POST /v1/sessions/<id>/cancel` flips a session-scoped
`AtomicBool`. The evaluator threads an `Interrupt` token through
`eval_program` and checks it at the head of every loop iteration
(`for`, `train`, `repeat`) plus before every builtin dispatch. On
trip, raises `EvalError::Cancelled` with the current step number
and the partial `last_losses`. `mlpl-repl --connect` binds Ctrl-C
on the second press to a cancel POST. Tests: cancel mid-`train`
returns the partial loss curve, cancel mid-builtin returns
promptly, double-cancel is idempotent.

### Phase 3 -- Visualization storage (2 steps)

#### Step 004 -- viz storage endpoint
`POST /v1/viz` stores an SVG / PNG / HTML / JSON payload and
returns a content-addressed URL (`/v1/viz/<sha256-prefix>`). `GET`
serves the bytes back with the right `Content-Type`. The server's
eval pipeline writes any returned viz value to viz storage *and*
to the local cache dir if `MLPL_CACHE_DIR` is set, returning the
URL in the eval response. `mlpl-repl --connect` prints both `viz:
<url>` and `viz: <local-path>` when both are present.

#### Step 005 -- non-SVG viz cache + format table
The `is_svg_string` detector grows into a small format table:
`<svg`, `<!DOCTYPE html`, magic-byte PNG/JPEG sniffing, and an
explicit `application/json` opt-in for traces. Extends both the
`MLPL_CACHE_DIR` path and the viz storage endpoint. Demo: capture
a `loss_curve(last_losses)` PNG, fetch it back via the URL, render
in a browser tab.

### Phase 4 -- Web UI re-routing (3 steps)

#### Step 006 -- web REPL gains a connect mode behind a feature flag
A new `?connect=<url>` query string (or a settings panel toggle)
switches the web REPL from the in-process WASM evaluator to the
remote one. Same evaluator entry point, two transports. The WASM
path stays the default. Tests: hand-rolled Playwright (or the
existing `mcp__playwright__*` tools) round-trip on `iota(5) + 1`
producing the same display in both modes.

#### Step 007 -- live training in the browser
The web REPL consumes the Phase 1 SSE endpoint when a `train { }`
runs in connect mode. The existing loss-curve UI piece updates in
place (line draws as metrics arrive). The cancel button hooks the
Phase 2 endpoint. Tutorial gains a "Train remotely" lesson.

#### Step 008 -- visualization rendering via storage URL
In connect mode the web REPL fetches viz bytes from the Phase 3
storage endpoint rather than expecting an inline payload. The
`<details>` numeric-summary accordion still renders inline; only
binary viz routes through storage.

### Phase 5 -- Persistence (2 steps)

#### Step 009 -- session re-attach across client restart
`mlpl-repl --connect --session <id> --token <tok>` rebinds to an
existing server-side session instead of creating a new one.
`/v1/sessions/<id>` (GET) returns the session metadata
(creation time, last-eval timestamp, vars / models / tokenizers
summaries). The token persists across REPL restarts via a small
keyring entry (or `MLPL_REPL_SESSION_FILE` env var for non-keyring
hosts).

#### Step 010 -- session persistence across server restart
Server-side: dump the `Environment` for every session to a
single-file SQLite DB on a configurable interval (`--persist
<path>` enables; default off). On startup, restore sessions from
that file. Tests: train a tiny model, restart the server, rebind
the client to the same session ID, `:vars` shows the same workspace
state. Out of scope: cross-machine migration.

### Phase 6 -- Wider wire dtypes (1 step)

#### Step 011 -- f32 + u8 over the MLX peer wire
`services/mlpl-mlx-serve/src/wire.rs` grows two dtype slots beyond
the current f64-only path: `DTYPE_F32 = 1`, `DTYPE_U8 = 2`. The
peer protocol uses the dtype the orchestrator sends; the
orchestrator picks dtype based on the source MLPL array. Image
tensors materialize as u8 at the source and stay u8 over the wire
until the first arithmetic op upgrades them. Tests: round-trip
f32 and u8 fixtures, parity vs f64 on a fixed-seed training step
(within tolerance).

### Phase 7 -- Docs + release (2 steps)

#### Step 012 -- using-cli-server.md retrospective + new client lessons
Move every Phase 1-6 item out of the "Non-goals (deferred)" list
into the main body of `docs/using-cli-server.md`. Add a "Streaming
and cancellation" section and a "Web UI in connect mode" section.
Two new web REPL tutorial lessons: "Connect to a remote MLPL
server" and "Long training, live loss". Update `docs/status.md`
one-liner and `docs/saga.md` Saga 21.5 entry.

#### Step 013 -- release v0.20.0
Bump workspace version. Tag `v0.20.0`. Push commit and tag.
Verify the pages workflow deploys the updated tutorial list and
the web REPL connect-mode toggle.

## Planned steps

| # | Slug | Phase | What it delivers |
|---|------|-------|------------------|
| 001 | sse-endpoint-scaffold        | 1 | `/eval_stream` SSE endpoint, metric/done/error events |
| 002 | sse-connect-client           | 1 | `mlpl-repl --connect --stream` consumes SSE |
| 003 | cancellation                 | 2 | `/cancel` + Interrupt token + Ctrl-Ctrl-C bind |
| 004 | viz-storage-endpoint         | 3 | `POST /v1/viz` + URL fetch, eval pipeline integration |
| 005 | viz-format-table             | 3 | PNG/HTML/JSON detection in cache + storage |
| 006 | web-repl-connect-mode        | 4 | `?connect=<url>` toggle, dual-transport evaluator |
| 007 | web-repl-streaming-train     | 4 | live loss curves over SSE in the browser |
| 008 | web-repl-viz-storage         | 4 | browser fetches viz bytes from the storage endpoint |
| 009 | session-reattach-client      | 5 | `--session <id> --token <tok>` reconnect |
| 010 | session-persist-server       | 5 | SQLite session persistence across restarts |
| 011 | wire-f32-u8                  | 6 | f32 / u8 dtypes on the MLX peer wire |
| 012 | docs-and-tutorials           | 7 | `using-cli-server.md` rewrite, two new lessons |
| 013 | release-v020                 | 7 | version bump, tag, deploy |

Thirteen steps. Phase 1 + 2 + 3 deliver a usable streaming server
on their own; if scope pressure hits, Phase 5 (persistence) is the
easiest to split into a Saga 21.6.

## Success criteria

- A `train 200 { }` block in `mlpl-repl --connect --stream` shows
  the loss decreasing line-by-line in the terminal.
- Ctrl-C twice during a long `train` cancels cleanly and prints
  the partial `last_losses` so far.
- `apps/mlpl-web` with `?connect=http://localhost:6464` runs the
  same `demos/tiny_lm.mlpl` source as the WASM path, produces the
  same final loss, and renders the loss curve as it trains.
- `mlpl-repl --connect --session <id> --token <tok>` re-attaches
  to a session after a client restart and sees the same `:vars`.
- A server restart with `--persist /tmp/sessions.db` brings the
  same `:vars` back.
- An image tensor materialized as u8 on the orchestrator stays
  u8 on the wire to `mlpl-mlx-serve` and is decoded to f32 (or
  promoted to f64) at the peer; the wire-size measurement is a
  recorded `cargo bench` line in `docs/benchmarks.md`.
- The existing offline-tutorial flow on `apps/mlpl-web`
  (WASM-only mode) keeps working with zero regressions.
- All eight items listed under "Non-goals (deferred)" in
  `docs/using-cli-server.md` (except LLM proxy and WebSocket) have
  moved into the main body or into a clearly-named follow-up.

## Risks and open questions

- **CORS for browser-against-server.** `apps/mlpl-web` running on
  `https://sw-ml-study.github.io/sw-mlpl/` cannot hit a localhost
  server without explicit CORS headers. Decide between
  ship-permissive-CORS (easy, slightly worrying) and ship
  `mlpl-serve` with a `--cors-allow <origin>` flag (safer, one
  more knob). Prefer the flag.
- **SSE through HTTP/2 proxies.** Some corporate proxies buffer
  SSE; document the mitigation (`X-Accel-Buffering: no`,
  flush-after-event) rather than chasing every quirk.
- **Cancellation cooperatively.** Pure-arithmetic builtins on
  large arrays don't yield. First version checks at op boundaries
  only; document the latency floor (one op = uncancellable). If
  this bites, add periodic checks inside the array loops as a
  follow-up.
- **SQLite for session persistence.** Workspace state includes
  models, tokenizers, experiments -- all serializable but not yet
  via a stable on-disk format. The simplest implementation
  serializes the in-memory `Environment` via serde + bincode and
  blobs it into one row per session. Schema migration becomes a
  problem at the first incompatible change; a `--persist-version`
  on the file is the cheap mitigation.
- **u8 wire dtype semantics.** On the peer, u8 stays u8 only as
  long as no arithmetic op touches it; the first `add` / `mul`
  promotes to f32 or f64. Specify the promotion ladder explicitly
  in the wire contract so client and peer agree.
- **Web UI dual-mode complexity.** Two transports doubles the
  test surface. Mitigation: a small evaluator-trait abstraction
  in `apps/mlpl-web/src/eval.rs` with two impls (WASM, REST), and
  one shared contract test that both impls satisfy on a fixture
  program.
- **LLM proxy is deferred again.** The proxy that lets browser
  `llm_call` reach a server-side allow-listed Ollama explicitly
  stays out of scope here -- it wants its own security review and
  the threat-model write-up that goes with that. A follow-up
  Saga 21.6 picks it up once the rest of Phase 1-7 has settled.
