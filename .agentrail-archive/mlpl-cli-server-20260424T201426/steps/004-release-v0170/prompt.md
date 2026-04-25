Phase 4 step 004: release v0.17.0.

1. **Version bump**. `Cargo.toml`
   workspace.package.version `0.16.0 -> 0.17.0`.
   Minor-level bump because the surface adds a new
   binary crate (`mlpl-serve`) plus a new
   `mlpl-repl --connect` client mode plus a CLI
   viz cache strategy. Not a patch.

2. **`CHANGELOG.md`** new v0.17.0 section above
   v0.16.0 (Saga 19's release):
   - Added: `crates/mlpl-serve` -- new binary +
     library crate exposing a REST + WebSocket
     surface over a long-running MLPL
     interpreter. MVP endpoints: `POST
     /v1/sessions` (create session + bearer
     token), `POST /v1/sessions/{id}/eval` (run a
     program against the session's env), `GET
     /v1/sessions/{id}/inspect` (workspace
     snapshot for client slash commands), `GET
     /v1/health`. Constant-time token compare;
     `--bind 0.0.0.0` requires `--auth
     required`. Contract:
     `contracts/serve-contract/sessions-and-
     eval.md`.
   - Added: `mlpl-repl --connect <url>` -- thin
     CLI client that delegates evaluation to a
     remote `mlpl-serve`. Local `Environment` is
     unused; slash commands fetch JSON snapshots
     via the new `/inspect` endpoint. Cannot
     combine with `-f` or `--data-dir`.
     Contract: `contracts/repl-contract/
     connect.md`.
   - Added: CLI visualization cache strategy.
     `mlpl-repl` (both local + connect modes)
     now writes returned SVG strings to
     `$MLPL_CACHE_DIR/<sha256-prefix>.svg`
     (default `~/.cache/mlpl/`) and prints
     `viz: <path>` instead of the raw `<svg>`
     XML. Module:
     `crates/mlpl-cli/src/viz_cache.rs`.
   - Added: `docs/using-cli-server.md`
     retrospective + user guide.
   - Changed: `apps/mlpl-repl/Cargo.toml` --
     `reqwest` (blocking, json) added for the
     `--connect` client.
   - Changed: `docs/configurations.md` -- CLI
     server column refreshed to reflect what is
     actually shipped vs. what remains post-MVP.
   - Tests: list counts for
     `crates/mlpl-serve/tests/api_tests.rs`,
     `apps/mlpl-repl/tests/connect_tests.rs`,
     `crates/mlpl-cli/tests/viz_cache_tests.rs`.
   - Scope notes: server-side LLM proxy with
     allow-list, visualization storage URLs,
     Server-Sent-Events streaming, cancellation,
     desktop GUI wrapper (tauri/wry), Emacs
     client, ratatui TUI, web UI re-routing to
     call origin -- all explicit non-goals
     deferred to follow-up sagas after the MVP
     server contract proves stable.

3. **`docs/saga.md`**. Saga 21 retrospective
   inserted above Saga 19 (the most recent
   retrospective) to preserve reverse-
   chronological ordering.

4. **`docs/status.md`**. Saga 21 row moves from
   Planned to Completed with `[x]` and the v0.17.0
   tag. "Next saga to start" pointer rewrites to:
   "(dev host move to Linux), then Saga 17 (CUDA
   + distributed). Post-MVP follow-ups (LLM
   proxy, SSE, desktop GUI, Emacs, web UI re-
   routing) fold into a follow-up CLI-server saga
   after the MVP proves stable in real use."

5. `cargo build --release` to confirm the bump.

6. `./scripts/build-pages.sh` to rebuild pages/ --
   docs/configurations.md changed; commit the new
   bundle.

7. `./scripts/gen-changes.sh` to refresh
   CHANGES.md; commit.

8. `/mw-cp` quality gates (cargo test, clippy,
   fmt, markdown-checker on changed files,
   sw-checklist baseline hold).

9. Tag `v0.17.0` locally. DO NOT push the tag
   without explicit user confirmation (v0.13.0 /
   v0.14.0 / v0.14.1 / v0.15.0 / v0.16.0 cadence).

10. `agentrail complete --done`.
