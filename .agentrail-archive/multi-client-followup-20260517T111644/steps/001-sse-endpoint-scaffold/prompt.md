Saga 21.5 step 001: sse-endpoint-scaffold.

Goal: ship a new `POST /v1/sessions/<id>/eval_stream` endpoint in
`mlpl-serve` that returns `text/event-stream` and round-trips an
`event: ready` -> zero-or-more `event: metric` -> terminal
`event: done` (or `event: error`) sequence for any program the
existing `/eval` endpoint accepts. The existing `/eval` MUST keep
working byte-for-byte; this step is additive.

TDD (Red/Green/Refactor):

1. RED: write failing integration tests in
   `crates/mlpl-serve/tests/sse_tests.rs` for:
   - `POST /v1/sessions/<id>/eval_stream` with `Authorization:
     Bearer <token>` and `{"program": "iota(5) + 1"}` returns:
       * HTTP 200
       * `Content-Type: text/event-stream`
       * an `event: ready` frame as the first event
       * a final `event: done` frame whose `data:` JSON payload
         matches the existing `/eval` response body verbatim
         (same `{"value": "...", "kind": "..."}` shape).
   - A program containing `train 10 { ... loss_metric = ... }`
     produces zero-or-more `event: metric` frames before the
     `event: done`, each frame's `data:` payload is JSON
     `{"name": "<metric_name>", "step": <i>, "value": <f64>}`,
     and the count + ordering matches the final `last_losses`
     vector.
   - Missing / invalid auth returns 401 (same posture as `/eval`).
   - Invalid session id returns 404.
   - A program that raises `EvalError` produces a terminal
     `event: error` frame with the same structured error body
     `/eval` returns.

   Reuse the existing test harness in `tests/api_tests.rs` for
   spinning up the server; consume SSE frames with a small
   helper that reads `Content-Type: text/event-stream` byte
   chunks and splits on the `\n\n` event delimiter.

2. GREEN: implement
   - new file `crates/mlpl-serve/src/sse.rs` housing:
       * `SseEvent` enum (Ready, Metric { name, step, value },
         Done(EvalValueBody), Error(StructuredError)).
       * `impl SseEvent` -> `axum::response::sse::Event`
         serialization (each variant -> one `event: <kind>` +
         `data: <json>` frame).
   - new handler `eval_stream_handler` in `handlers.rs`:
       * auth + session lookup identical to `eval_handler`.
       * Creates a `tokio::sync::mpsc::channel::<SseEvent>` and
         returns an `axum::response::sse::Sse` stream over the
         receiver, with keep-alive enabled (the existing axum
         dependency already supports `sse` via `Sse::new`).
       * Pushes `SseEvent::Ready`, then runs the eval on a
         `tokio::task::spawn_blocking` task to keep the runtime
         healthy; the eval task scans the env for keys ending
         in `_metric` after each `train { }` iteration and
         emits one `Metric` event per scalar (see "Metric
         capture mechanism" below).
       * Pushes `SseEvent::Done(...)` or `SseEvent::Error(...)`
         at exit.
   - Route registration in `server.rs`: add
     `.route("/v1/sessions/:id/eval_stream", post(handlers::
     eval_stream_handler))` alongside the existing `/eval`
     route.

   Metric capture mechanism: add a `MetricSink` trait to
   `mlpl-eval` (or a closure-typed field on `Environment`) so
   `eval_train` can call back into the server on every loop
   iteration with `(name, step, value)` triples for every
   binding that ends in `_metric`. Default is a no-op so all
   existing callers stay untouched. The server's
   eval_stream_handler installs a sink that pushes to the
   channel.

3. REFACTOR: keep `crates/mlpl-serve/src/` modules under the
   sw-checklist 7-fn budget. `sse.rs` holds the event type,
   serialization, and any helpers; the channel plumbing lives
   in `handlers.rs` alongside the existing eval_handler.

Contract update (REQUIRED in same commit):
- Append a new section "SSE streaming eval (`/eval_stream`)"
  to `contracts/serve-contract/sessions-and-eval.md` with:
  * endpoint + method + auth + path param shape
  * event-kind enumeration: ready / metric / done / error
  * `data:` payload schema per event kind (with examples)
  * ordering guarantees (ready always first, exactly one
    done-or-error terminal frame)
  * the explicit statement that `/eval` semantics are
    unchanged.

Quality gates per /mw-cp (CLAUDE.md):
- `cargo test -p mlpl-serve` first (fast inner loop)
- `cargo test` full workspace before commit
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo fmt --all` and `cargo fmt --all -- --check`
- `markdown-checker -f "contracts/serve-contract/sessions-
  and-eval.md"` (docs touched)
- `sw-checklist` (baseline 106 passed / 134 failed must hold)

Commit before `agentrail complete`. Push after commit.

Out of scope for this step (lands in later steps):
- `mlpl-repl --connect --stream` client flag (step 002).
- Cancellation (step 003).
- Visualization storage URL integration (step 004).
- CORS headers for browser-from-Pages access (Phase 4).
- Replacing `/eval` with `/eval_stream` -- both endpoints
  remain peers indefinitely.
