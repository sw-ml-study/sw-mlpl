Saga 21.5 step 010: session-persist-server.

Goal: mlpl-serve persists the Environment for every session to a single-file SQLite DB on a configurable interval (--persist [path] enables, default off). On startup, restores sessions from that file. Tests: train a tiny model, restart the server, rebind the client to the same session ID, :vars shows the same workspace state. Cross-machine migration is out of scope.

TDD (Red/Green/Refactor):

1. RED tests in crates/mlpl-serve/tests/persistence_tests.rs:
   - Train and bind some vars in a session.
   - Trigger a save (via an interval tick or a programmatic flush helper).
   - Spin up a NEW mlpl-serve instance pointing at the same DB file.
   - Issue GET /v1/sessions/[id] with the original token -- expect the saved created_at, last_eval_at, vars, etc.

2. GREEN:
   - New crates/mlpl-serve/src/persist.rs (or inline mod in lib.rs to defend module count) wrapping rusqlite. One row per session: session_id, token, created_at, last_eval_at, env_blob_bincode.
   - Environment must be serde-serializable. Add serde derives where missing (Environment + ModelSpec + TokenizerSpec + Value variants the env can hold). bincode + serde for compact on-disk format.
   - --persist [path] flag: sets the DB path; absent means in-memory (default behavior preserved).
   - --persist-interval-secs [N] flag: default 30; background tokio task flushes dirty sessions every N seconds. Also flush on graceful shutdown.
   - Startup load: reads all rows from the DB and populates the SessionMap before the listener accepts.
   - Schema migration: add a persist_version column; future incompatible changes bump it.

3. REFACTOR: keep sw-checklist budgets; the persist module is the natural home for save / load / flush_dirty / schema_init. mlpl-serve crate module count is at 8 (FAIL line); adding 1 more is same FAIL line.

Quality gates per /mw-cp: cargo test, cargo clippy, cargo fmt, markdown-checker, sw-checklist. Update contract: new --persist + --persist-interval-secs flags, durability semantics, schema-version note.

Out of scope: f32/u8 MLX peer wire (step 011); docs + release (012/013); cross-machine migration; encrypted-at-rest storage.