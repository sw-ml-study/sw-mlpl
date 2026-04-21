# Web REPL: Web Worker architecture (plan)

> Status: planning doc. Not yet implemented. Opens follow-up work
> tracked in `docs/plan.md` as a post-v0.11 web-UX item.

The web REPL at `sw-ml-study.github.io/sw-mlpl` runs every `eval`
call synchronously on the browser's main thread. That is fine
for small ops and the tutorial lessons, but it breaks down for
training-loop demos: the browser's "unresponsive tab" dialog
fires at roughly 15-30 seconds of blocked main-thread time, and
a `train 200 { adam(...) }` line on the Saga 13 Tiny LM
configuration easily exceeds that under the WASM interpreter.

Two half-measures already shipped after the initial freeze
report:

1. **Tutorial-budget demo configs.** `Tiny LM` and `Tiny LM
   Generate` in `apps/mlpl-web/src/demos.rs` now match the
   tutorial lesson's budget (V=260, d=16, block=8, 30 steps)
   rather than the CLI-oriented `demos/tiny_lm.mlpl` budget
   (V=280, d=32, 200 steps). Completes in a few seconds.
2. **Cross-line yield.** `make_run_demo` in
   `apps/mlpl-web/src/handlers.rs` now schedules each demo line
   in its own `gloo::timers::callback::Timeout::new(0, ...)`
   tick instead of a synchronous for loop. The browser can paint
   and process input *between* lines.

Both are correct and ship today. Neither changes the fact that
a single long-running line (a `train N { ... }` loop, a large
batched forward, etc.) still blocks the main thread for the
duration of that single line. The proper fix is to move `eval`
off the main thread entirely: put the interpreter in a Web
Worker, message lines across the boundary, and render results
as they arrive.

## Goals

- **Main thread stays responsive.** No single MLPL line, no
  matter how expensive, triggers the browser's unresponsive
  dialog. Scrolling, input, button clicks all work while a
  training loop runs.
- **Cancellable runs.** Every demo and every long `train { }`
  can be aborted with a visible cancel button. Today there is
  no way to stop a runaway eval short of closing the tab.
- **Streaming output.** Intermediate values (e.g. per-step
  loss during `train { }`) arrive in the REPL history as they
  are produced, not in one batch at the end. Makes the tab
  feel alive and makes failures diagnosable.
- **Same `WasmSession` surface.** The `mlpl-wasm::WasmSession`
  API that `handlers.rs` uses today stays the primary public
  interface; the worker is an implementation detail hidden
  behind a thin async wrapper.
- **No regression for the tutorial.** Single-line inputs from
  the REPL prompt keep working identically. The latency cost
  of crossing the worker boundary (typically < 1 ms per
  message) is not visible to the user.

## Non-goals

- **No parallelism inside the interpreter.** WASM threads on
  the web are available (`SharedArrayBuffer` + COOP/COEP
  headers) but the Saga 13 interpreter has no shared-state
  hazards worth the shipping complexity. One worker =
  single-threaded interpreter, same as today -- we only move
  the thread it runs on.
- **No architectural rewrite of `mlpl-eval`.** The interpreter
  keeps its current APIs. The worker crate wraps `mlpl-wasm`
  behind a message-passing shim.
- **No serialized `DenseArray` wire format beyond what
  `WasmSession::eval` already returns as a string.** If the
  worker needs to send richer values (Trace JSON, raw array
  bytes for large viz), that is a separate follow-up -- this
  plan keeps string in, string out.
- **No dependency on a rebuilt stdlib.** Rust stable targeting
  `wasm32-unknown-unknown` with Yew + Trunk already supports
  Web Workers via `wasm-bindgen` and
  `gloo_worker`. No nightly features required.

## Architecture

Three layers:

```
                                      worker thread
  +--------------------+  spawn_local  +-------------------+
  | yew components     | ----------->  | gloo_worker::     |
  | handlers.rs        |               |   Registrable impl|
  |                    |  .send(msg)   |                   |
  |                    | ------------> |   message pump    |
  |                    |               |   |               |
  |                    |  .callback    |   v               |
  |                    | <------------ |   mlpl-wasm       |
  |                    |               |     WasmSession   |
  +--------------------+               +-------------------+
        main thread                      DedicatedWorker
```

### Worker crate

A new `crates/mlpl-wasm-worker/` crate. Exposes a
`gloo_worker::Worker` implementation whose `Reach` is
`DedicatedWorker` and whose:

- `Input` is `enum EvalRequest { Eval { id: u64, src: String },
  Cancel { id: u64 }, Clear, Reset }`.
- `Output` is `enum EvalEvent { Started { id: u64 }, Partial {
  id: u64, line_output: String }, Completed { id: u64, final_
  output: String, is_error: bool }, Cancelled { id: u64 },
  Warning { message: String } }`.
- Internal state: a single `mlpl_wasm::WasmSession`. One
  session per worker, long-lived across messages -- matches
  today's main-thread model where the REPL session persists
  across inputs.

Every `Eval` request runs to completion on the worker; the
worker posts `Started` first, may post one or more `Partial`
events as the eval progresses, and ends with `Completed` or
`Cancelled`. The `id` threads through so the UI can match
events back to the originating request and discard stale
events after a cancel.

### Cancellation

The worker checks a cooperative cancel flag at known yield
points in the interpreter. The two that matter:

1. `Expr::Train { .. }` -- between training steps. A 100-step
   loop that takes 10 seconds total should cancel within one
   step (~100 ms).
2. `Expr::Repeat { .. }` -- between iterations. Same pattern.

Finer-grained cancellation inside a single op (a large matmul,
a cross_entropy over a huge batch) is not in scope for the
first worker landing; those are rare in the demo workloads and
the tutorial-budget configs already avoid them.

Implementing this needs a small patch to `mlpl-eval::eval`:
pipe an `Arc<AtomicBool>` (or a `CancellationToken`) through
the `Train` and `Repeat` arms, check it once per iteration,
raise a new `EvalError::Cancelled` when set. Cost: maybe 50
LOC in `mlpl-eval` + passing the flag through `Environment` or
a separate context handle.

### Streaming output

The hook the worker cares about is `train { ... }`'s per-step
loss capture. Today `train N { body }` collects the
`loss_metric` and every body-last-value into `last_losses`
after the loop completes; the web UI only sees the completed
result. For streaming, the worker would:

1. After each training step completes inside the worker, read
   the just-appended `last_losses[step]` value.
2. Post a `Partial { id, line_output: format!("step={step}
   loss={loss}") }` event.
3. The main-thread handler appends to the history entry for
   the current line rather than replacing it.

This needs a callback slot on `train`'s evaluator to fire
after each step. ~30 LOC in `mlpl-eval`, plus a small wasm-
side closure that forwards to the worker output channel.

### Message flow

Normal single-line REPL input:

```
user types "iota(5)"
  main: session_proxy.eval("iota(5)", id=N)
  worker: recv EvalRequest::Eval { id=N, src="iota(5)" }
  worker: post EvalEvent::Started { id=N }
  worker: run session.eval("iota(5)")
  worker: post EvalEvent::Completed { id=N, final_output: "[0, 1, 2, 3, 4]", is_error: false }
  main: append history entry, clear input
```

Train loop with streaming:

```
user clicks "Run demo: Tiny LM"
  main: for each line in demo.lines, session_proxy.eval(line, id=K..K+n)
  worker: recv Eval(id=K, "corpus = load_preloaded(...)")
  worker: Completed(id=K, "...")
  worker: recv Eval(id=K+1, ...)
  ...
  worker: recv Eval(id=K+9, "experiment \"tiny_lm\" { train 30 { ... } }")
  worker: Started(id=K+9)
  worker: per step, Partial(id=K+9, "step=0 loss=5.2")
  worker: ... 30 events ...
  worker: Completed(id=K+9, "...")
  main: each event appends to or finalizes the history row
```

User hits Cancel while training:

```
  main: session_proxy.cancel(id=K+9)
  worker: recv Cancel(id=K+9)
  worker: sets its AtomicBool flag
  worker: Expr::Train's next step-boundary check sees the flag, bails
  worker: post EvalEvent::Cancelled { id=K+9 }
  main: show "cancelled" in the history row, re-enable the input
```

### Session reset + demo runs

`WasmSession::clear()` today lives on the main thread and is
fire-and-forget. In the worker model it becomes a message
(`EvalRequest::Clear`) that the worker processes in order with
`Eval` requests -- so the existing `handlers.rs` "clear + run
every demo line" pattern maps cleanly to "Clear, then Eval per
line" without a race.

## Crate layout

```
crates/mlpl-wasm/              (exists today -- the Session + bindings)
crates/mlpl-wasm-worker/       (new -- gloo_worker impl + message types)
apps/mlpl-web/
  src/handlers.rs              (rewire to use the proxy + async callbacks)
  src/worker_bridge.rs         (new -- thin wrapper around gloo_worker Bridge)
```

`mlpl-wasm-worker` re-exports the `EvalRequest` / `EvalEvent`
enums so both main-thread code and the worker binary see the
same message types. The web app gets one additional asset
(`mlpl_worker_bg.wasm`) shipped via Trunk's worker support.

## Migration plan

Five small steps so the UI stays shippable at every commit:

1. **Add `mlpl-wasm-worker` crate with no UI integration.** The
   worker compiles and is tested via headless wasm tests
   (`wasm-bindgen-test`). No change to the live UI.
2. **Wire a `worker_bridge` module in `apps/mlpl-web`.** Opens
   a worker on app start, exposes an async `eval_on_worker`
   helper. Not yet used.
3. **Migrate REPL input to the worker.** `handlers.rs`
   `make_submit` switches from `session.borrow().eval(...)` to
   `eval_on_worker(...)` with a yield loop. Tutorial lessons
   still work; demos still use the sync path. Ships as the
   first visible improvement.
4. **Migrate demos to the worker + cancel button.** `make_run_
   demo` schedules via `eval_on_worker`; a new "Cancel run"
   button in the UI sends `EvalRequest::Cancel`. Requires the
   `mlpl-eval` cancellation-flag plumbing described above.
5. **Add training-step streaming.** `mlpl-eval` grows a
   per-step callback; the worker forwards step events as
   `EvalEvent::Partial`; the UI renders a live loss readout.

Each step is a coherent ship -- gate `/mw-cp`, write a
migration test harness (mirrors `mlpl-wasm`'s existing tests),
push, commit.

## Risks and mitigations

- **Worker bundle size.** Each worker is its own `.wasm`
  binary. `mlpl-wasm-worker` should depend only on
  `mlpl-eval`, `mlpl-parser`, `mlpl-runtime`, `mlpl-viz` --
  not on Yew. Trunk's worker build keeps the UI chunk and the
  worker chunk separate.
- **Shared-memory temptation.** WASM threads + `SharedArray
  Buffer` require cross-origin isolation headers
  (COOP/COEP). GitHub Pages serves neither. A single-worker
  design avoids the whole question -- we get off the main
  thread without needing shared memory.
- **Message round-trip overhead.** Every `eval` becomes an
  async round trip. Measured cost of `postMessage` with small
  string payloads is well under 1 ms; this is the right trade
  for the UX win. If it ever becomes noticeable, batch
  multiple small inputs into one request.
- **State-split bugs.** Putting the `WasmSession` in the
  worker means the main thread cannot introspect variables
  directly for autocomplete or `:vars`-style output. The
  existing `:vars` / `:describe` REPL commands already flow
  through `eval`, so they keep working; any future feature
  that would have peeked at session state directly must send
  a message.
- **Panic recovery.** A panic inside the worker today kills
  the worker. The main-thread proxy should detect
  `onerror`/closed-port conditions, show a "worker crashed,
  reloading" notice, and spin up a fresh worker with an empty
  session. One-line change using `gloo_worker::Spawnable`.

## Success criteria

When this lands:

- `Tiny LM` and `Tiny LM Generate` demos with the tutorial-
  budget configs run to completion with the browser staying
  fully responsive throughout. Scrolling works; the input box
  stays focusable; switching tabs and coming back shows
  continued progress.
- A visible Cancel button stops any running eval within one
  training-step's worth of work (~100 ms for Tiny LM, well
  under a second).
- Per-step loss values appear in the REPL history as training
  runs, not just after it completes.
- `cargo test` workspace clean, `cargo clippy` clean,
  `sw-checklist` FAIL count unchanged, new tests land for
  `mlpl-wasm-worker`'s message round-trip.

## Related

- `apps/mlpl-web/src/handlers.rs` -- current main-thread
  demo runner with the cross-line yield shipped today
- `apps/mlpl-web/src/demos.rs` -- demo configs (now matching
  the tutorial budget on the LM entries)
- `crates/mlpl-wasm/` -- the `WasmSession` that will move
  into the worker wholesale
- `crates/mlpl-eval/` -- needs the cancellation flag + per-
  step callback hooks described above
- `docs/using-mlx.md` -- MLX dispatch is unrelated to the
  worker story: WASM cannot dispatch to MLX, so this doc's
  scope is the CPU interpreter path only. The worker plan
  stacks cleanly on top of MLX once a native desktop UI
  lands (then the worker is an `async fn` instead of a
  postMessage bridge; same contract).
