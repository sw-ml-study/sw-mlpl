Saga 21.5 step 009: session-reattach-client.

Goal: mlpl-repl --connect --session <id> --token <tok> rebinds to an existing server-side session instead of creating a new one. Two pieces:

1. Server: GET /v1/sessions/<id> returns session metadata (creation_time, last_eval_at, vars/models/tokenizers/experiments summaries) for clients to display before resuming.
2. Client: --session <id> --token <tok> flags on mlpl-repl --connect that bypass the create-session call. Token persists across REPL restarts via either a keyring entry (best-effort) or an MLPL_REPL_SESSION_FILE env var fallback for hosts without a keyring.

TDD (Red/Green/Refactor):

1. RED tests:
   - crates/mlpl-serve/tests/session_meta_tests.rs (new): GET /v1/sessions/<id> returns metadata for an existing session; 404 for unknown id; 401 without bearer; metadata reflects bound vars after an eval.
   - apps/mlpl-repl/tests/connect_reattach_tests.rs (new): pure-fn parser for --session + --token flags; reattach skips create-session and uses the provided id/token.

2. GREEN:
   - crates/mlpl-serve: GET /v1/sessions/:id handler (alongside existing POST /v1/sessions). Returns JSON { session_id, created_at, last_eval_at, vars, models, tokenizers, experiments, more }. Session struct grows created_at + last_eval_at fields populated by create_session_handler and updated by eval_handler. Contract update.
   - apps/mlpl-repl: parse_connect_args grows session/token flags. connect_repl::read_loop accepts (id, token) directly when both are passed. Validates the pair by GET /v1/sessions/<id> before printing the welcome banner.
   - Token persistence: minimal MLPL_REPL_SESSION_FILE env var path that reads/writes a TOML or JSON file with {session_id, token}; keyring is out of scope (a follow-up).

3. REFACTOR: keep sw-checklist budgets; mlpl-serve handlers.rs is at 7 fns max -- session_meta_handler may need to go in sessions.rs or as inline submod in lib.rs.

Quality gates per /mw-cp: cargo test, cargo clippy, cargo fmt, markdown-checker, sw-checklist (held). Commit before agentrail complete.

Out of scope: server-side persistence across restart (step 010 = SQLite). Keyring-backed token storage (follow-up).