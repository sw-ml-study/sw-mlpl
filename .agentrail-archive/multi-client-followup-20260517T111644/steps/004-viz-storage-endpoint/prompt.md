Saga 21.5 step 004: viz-storage-endpoint.

Goal: ship POST /v1/viz (stores SVG/PNG/HTML/JSON payload, returns a content-addressed URL /v1/viz/<sha256-prefix>) + GET /v1/viz/<id> (serves bytes back with the correct Content-Type). The server's eval pipeline writes any returned viz value to BOTH viz storage AND the local MLPL_CACHE_DIR (when set), returning the URL in the eval response. mlpl-repl --connect prints 'viz: <url>' and 'viz: <local-path>' when both are present.

TDD (Red/Green/Refactor):

1. RED tests in crates/mlpl-serve/tests/viz_storage_tests.rs:
   - POST /v1/viz with an SVG payload returns 200 + {url: '/v1/viz/<hex>'}.
   - GET /v1/viz/<hex> returns the same bytes with Content-Type: image/svg+xml.
   - GET /v1/viz/<unknown-hex> is 404.
   - Auth-required mode: missing bearer is 401.
   - eval that returns an SVG-shaped string now includes a  field in /eval response.

2. GREEN:
   - New module (or inline in handlers.rs if budget permits) for viz_storage.
   - In-memory map keyed by sha256-prefix to bytes + content-type.
   - POST /v1/viz handler: accepts JSON {bytes_base64, content_type} or raw multipart; computes sha256, returns URL.
   - GET /v1/viz/<id> handler: serves bytes back.
   - eval/eval_stream pipeline integration: after eval_program_value, if the value is a string that the existing is_svg_string detector identifies as an SVG, write to viz storage AND MLPL_CACHE_DIR; return viz_url alongside value/kind.
   - mlpl-repl connect.rs: extend EvalResponse to include optional viz_url; print 'viz: <url>' when present.

3. REFACTOR: keep sw-checklist budgets. mlpl-serve is already at the 7-module ceiling -- inline the storage in an existing module (handlers.rs or a new pub mod inside lib.rs) rather than adding a top-level module file.

Quality gates per /mw-cp: cargo test (workspace), cargo clippy --all-targets --all-features -- -D warnings, cargo fmt + check, markdown-checker on touched contracts, sw-checklist (baseline must hold). Update contracts/serve-contract/sessions-and-eval.md with the new endpoints. Commit before agentrail complete. Push after commit.

Out of scope (later steps): non-SVG viz cache format table (step 005), web REPL connect mode (Phase 4).