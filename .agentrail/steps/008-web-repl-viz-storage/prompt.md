Saga 21.5 step 008: web-repl-viz-storage.

Goal: in connect mode the web REPL fetches viz bytes from the Phase 3 /v1/viz/<id> storage endpoint instead of expecting an inline payload. The existing numeric-summary <details> accordion (Saga 8 area) keeps rendering inline; only binary viz (SVG / PNG / HTML / JSON via VizFormat::detect) routes through the URL fetch.

TDD (Red/Green/Refactor):

1. RED tests:
   - apps/mlpl-web/tests/connect_viz_fetch_tests.rs (new): native test that drives RemoteEvaluator's response through a helper that recognizes the viz_url field, fetches the bytes, and surfaces them as an embeddable artifact (data URL or owned bytes). Mirrors the server-side eval pipeline integration tested in step 004.
   - Tests: SVG-returning eval populates viz_url on EvalResponse client-side; fetching the URL returns the same bytes with the right Content-Type; non-SVG responses don't trigger a fetch.

2. GREEN:
   - apps/mlpl-web/src/eval.rs (or new src/eval_viz.rs): the existing client EvalResponse already mirrors viz_url + viz_local_path (added in step 006). Add a RemoteEvaluator::fetch_viz(url) -> Future of (bytes, content_type) helper. Native impl uses reqwest::blocking; WASM impl uses gloo::net.
   - Wire viz display into the yew REPL: when an EvalResponse carries viz_url, fetch it and render as an <img> for image/* or an <iframe srcdoc> for text/html. Reuse the existing numeric-summary path for non-viz strings.

3. REFACTOR: split eval.rs into a sub-module tree if the file LOC stays a FAIL (eval/mod.rs + eval/remote.rs + eval/sse.rs + eval/url.rs); the step 007 exception specifically called this out as the natural follow-up cleanup.

Quality gates per /mw-cp: cargo test, cargo clippy, cargo fmt, markdown-checker, sw-checklist (pay down the step-007 exception by splitting eval.rs). scripts/build-pages.sh before commit.

Out of scope: session re-attach across client restart (step 009).