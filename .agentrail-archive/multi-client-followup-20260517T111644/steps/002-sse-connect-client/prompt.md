Saga 21.5 step 002: sse-connect-client.

Goal: `mlpl-repl --connect <url> --stream` (and the
`MLPL_REPL_STREAM=1` env var) routes every eval through the new
`/v1/sessions/<id>/eval_stream` endpoint from step 001. Each
`event: metric` frame redraws the same one-line loss display the
local REPL already uses for `train { }`. The default (no
`--stream` flag, no env var) keeps using `/eval` so existing
scripts stay quiet.

TDD (Red/Green/Refactor):

1. RED: write failing tests in
   `apps/mlpl-repl/tests/connect_stream_tests.rs` for:
   - `--connect <url>` with `--stream` produces the same final
     value as without `--stream` for `iota(5) + 1`.
   - A `train 5 { loss_metric = step + 0.5 }` program prints
     one progress line per iteration when `--stream` is set
     (test captures stdout from the repl driver).
   - `MLPL_REPL_STREAM=1` is equivalent to passing `--stream`.
   - `--stream` without `--connect` errors with exit code 2
     ("`--stream` requires `--connect`").

2. GREEN: implement
   - new `--stream` flag in
     `apps/mlpl-repl/src/cli.rs` (or wherever connect-mode flags
     live), reading `MLPL_REPL_STREAM` as the env fallback.
   - new module `apps/mlpl-repl/src/connect_stream.rs` (or
     extend the existing connect-mode client) that:
       * POSTs to `/v1/sessions/<id>/eval_stream` with bearer
         token.
       * Consumes the SSE body as a streaming response using the
         `eventsource-client` crate or a hand-rolled
         `text/event-stream` parser (Saga 21 step 002 already
         pulls in `reqwest` async).
       * On each `event: metric`, calls back into the existing
         per-step display function so the loss line redraws.
       * On `event: done`, prints the final value.
       * On `event: error`, prints the error and exits non-zero.
   - wire `--stream` so passing it routes connect-mode eval
     through this new path instead of the existing
     non-streaming POST.

3. REFACTOR: keep `mlpl-repl/src/connect.rs` (or the renamed
   module) under the sw-checklist 7-fn budget. Likely needs an
   extracted `render_metric` helper shared between streaming
   and non-streaming paths.

Quality gates per /mw-cp (CLAUDE.md):
- `cargo test -p mlpl-repl`
- `cargo test` workspace
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo fmt --all` and `cargo fmt --all -- --check`
- `markdown-checker -f "docs/using-cli-server.md"` (docs touched
  if the section moves from non-goals to main body)
- `sw-checklist` (baseline must hold)

Commit before `agentrail complete`. Push after commit.

Out of scope (later steps):
- Cancellation endpoint + Ctrl-C bind (step 003).
- Viz storage URLs (step 004).
- Web REPL connect mode (Phase 4).
