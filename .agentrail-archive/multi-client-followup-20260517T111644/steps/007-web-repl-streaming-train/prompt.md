Saga 21.5 step 007: web-repl-streaming-train.

Goal: the web REPL consumes the Phase 1 SSE endpoint when a train { } runs in connect mode. The existing loss-curve UI piece (or a new one) updates in place as metric frames arrive. The cancel button hooks the Phase 2 /cancel endpoint. Tutorial gains a Train remotely lesson.

TDD (Red/Green/Refactor):

1. RED tests:
   - apps/mlpl-web/tests/streaming_train_tests.rs (new): native test of a RemoteEvaluator::eval_stream method that POSTs to /eval_stream, parses SSE frames (ready/metric/done/cancelled/error), calls back on each metric. Mirrors apps/mlpl-repl/src/connect_stream.rs's eval_remote_stream helper.
   - Tests cover: 5-iter train emits 5 metrics; train 50000 + cancel-mid-flight surfaces partial losses; same-final-value parity with non-streaming /eval.

2. GREEN:
   - apps/mlpl-web/src/eval.rs (extend or split): RemoteEvaluator gains eval_stream + cancel methods. Native impl uses reqwest::blocking Response + BufReader::lines like mlpl-repl's connect_stream.rs. WASM impl uses gloo::net::http::Request + readable-stream + manual SSE frame parsing.
   - Wire the yew REPL flow: when ?connect=<url> + a 'train' program is detected, route through eval_stream and update a loss-curve UI piece in place (Yew use_state + use_effect).
   - Cancel button: existing tutorial UI or a new affordance triggers RemoteEvaluator::cancel which POSTs /cancel.
   - Tutorial: new 'Train remotely' lesson in apps/mlpl-web/src/lessons.rs.

3. REFACTOR: keep eval.rs sub-module structure clean. Document in contract that connect mode covers train { } streaming + cancel. Pages/ rebuild required.

Quality gates per /mw-cp: cargo test (workspace), cargo clippy, cargo fmt, markdown-checker, sw-checklist (held vs parent 136 OR pay down). scripts/build-pages.sh before commit.

Out of scope: viz storage fetch from browser (step 008).