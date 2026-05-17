Saga 21.5 step 006: web-repl-connect-mode.

Goal: the apps/mlpl-web WASM REPL gains a connect mode behind a feature flag. A new ?connect=<url> query string (or a settings panel toggle) switches the web REPL from the in-process WASM evaluator to the remote one through mlpl-serve. Same evaluator entry point, two transports. The WASM path stays the default. Tests: hand-rolled Playwright (or the existing mcp__playwright__* tools) round-trip on iota(5)+1 producing the same display in both modes.

TDD (Red/Green/Refactor):

1. RED tests:
   - apps/mlpl-web/tests/connect_mode_tests.rs (new): unit-test a small evaluator-trait abstraction in apps/mlpl-web/src/eval.rs with two impls (WASM, REST) -- both produce the same display for iota(5)+1 on a fixture program.
   - Playwright (mcp__playwright__*) integration: navigate to apps/mlpl-web with ?connect=http://127.0.0.1:PORT, type iota(5)+1, expect the same printed value as the WASM mode. Spin mlpl-serve up on a random localhost port in the test fixture.

2. GREEN:
   - apps/mlpl-web/src/eval.rs (new) with Evaluator trait (eval_string -> Result<String, EvalError>) and two impls: WasmEvaluator (existing in-process path) and RemoteEvaluator (POSTs to /v1/sessions/<id>/eval).
   - Main yew app parses ?connect=<url> from window.location.search; if set, build a RemoteEvaluator (via reqwest WASM target or web-sys::fetch); otherwise default to WasmEvaluator.
   - All Eval call sites in main.rs / handlers.rs route through the trait.

3. REFACTOR: keep evaluator-trait small (1-2 methods). CORS: mlpl-serve needs a --cors-allow <origin> flag so the browser fetch to a different origin succeeds (loopback default is fine). Document the flag + the dual-mode UX in contracts/serve-contract/sessions-and-eval.md.

Quality gates per /mw-cp: cargo test (workspace), cargo clippy, cargo fmt, markdown-checker on contract touch, sw-checklist (held). Web UI changes rebuild pages/ via scripts/build-pages.sh. Commit before agentrail complete; push after.

Out of scope: streaming SSE in browser (step 007); viz storage fetch from browser (step 008).