Phase 2 step 002: `mlpl-repl --connect <url>`
client + `GET /v1/sessions/{id}/inspect` server
endpoint.

With the server skeleton landed in step 001, ship
the first real client: a CLI REPL that delegates
all evaluation to a remote `mlpl-serve`. The local
`Environment` is unused -- the server holds all
state. Slash commands (`:vars`, `:models`, ...)
stay client-side but operate on a server-fetched
JSON snapshot.

1. **Flag**. New `--connect <url>` flag on
   `mlpl-repl`. Cannot combine with `-f` or
   `--data-dir` (those are local-mode-only); the
   parser errors helpfully if combined ("--connect
   delegates evaluation to a remote server; -f
   and --data-dir are local-mode only").

2. **Behavior**. With `--connect <url>` set:
   - On startup: POST `<url>/v1/sessions`. Store
     the returned `(session_id, token)` for the
     duration of the REPL.
   - For each user line: POST
     `<url>/v1/sessions/{id}/eval` with
     `{program}`. Print the returned `value` (or
     the `error` message). The local
     `mlpl_eval::Environment` is NOT consulted.
   - Slash commands -- `:vars`, `:models`,
     `:experiments`, `:describe` -- call the new
     `GET /v1/sessions/{id}/inspect` endpoint
     and render the JSON snapshot locally. The
     existing `apps/mlpl-repl/src/repl.rs`
     dispatcher branches on connect-vs-local.
   - `:ask` keeps using
     `mlpl_runtime::call_ollama` directly (it
     does not need server help; the local
     OLLAMA_HOST env var still works in connect
     mode). Document this in the contract.

3. **New server endpoint**. `GET /v1/sessions/
   {id}/inspect` (auth required). Returns:
   ```json
   {
     "vars": [{"name": "x", "shape": [3, 4],
                "is_param": false}, ...],
     "models": ["m"],
     "tokenizers": ["t"],
     "experiments": ["sweep-v1"]
   }
   ```
   Variables capped at 200 entries (server returns
   `more: <count>` when truncated). The endpoint
   shape mirrors what `:vars` / `:models` /
   `:experiments` print today; the client
   formatter just consumes JSON instead of walking
   `Environment` directly.

4. **Module layout**.
   - `apps/mlpl-repl/src/connect.rs` -- new
     module. Functions: `create_session`,
     `eval_remote`, `inspect_remote`,
     `read_loop`. Stay under 7-fn budget.
   - `apps/mlpl-repl/Cargo.toml` -- add `reqwest`
     (sync API: `reqwest = { version = "0.12",
     default-features = false, features =
     ["json", "blocking"] }` to avoid pulling
     tokio into the REPL crate).
   - `crates/mlpl-serve/src/handlers.rs` -- new
     `inspect_handler`. Stay under 7-fn budget;
     extract a private helper for the
     env-snapshot construction if needed.

5. **Tests**.
   - `apps/mlpl-repl/tests/connect_tests.rs` --
     spin up `mlpl-serve` in-process via
     `mlpl_serve::server::run(...)`, call
     `connect::create_session` + `eval_remote`
     directly (not through the REPL line
     reader -- that's hard to drive
     programmatically). Assert: session created,
     a few evals run remotely returning expected
     results, an error eval returns the server's
     error message.
   - `crates/mlpl-serve/tests/api_tests.rs` --
     extend with `GET /v1/sessions/{id}/inspect`
     happy path (set a var, inspect, see it in
     vars list), auth required (no bearer ->
     401), unknown session -> 404.

6. **Contract**.
   - `contracts/serve-contract/sessions-and-
     eval.md` -- add the new `inspect` endpoint
     section.
   - New `contracts/repl-contract/connect.md` --
     document `--connect`, the local-mode-only
     flag interaction, the slash-command JSON
     pattern, the `:ask` carve-out.

7. Quality gates + commit. Commit message
   references Saga 21 step 002.
