Phase 3 step 003: CLI viz cache strategy +
`docs/using-cli-server.md` retrospective + user
guide + `docs/configurations.md` refresh.

Today `mlpl-repl` (and `--connect` from step 002)
prints raw `<svg>...</svg>` XML inline whenever a
viz primitive returns a `Value::Str` SVG. That's
unusable in a terminal. Replace it with a cache-
dir + path-print strategy that works the same in
both local and connect modes. Then write the user-
facing docs for the whole saga.

1. **CLI viz cache strategy**.
   - New module `crates/mlpl-cli/src/viz_cache.rs`
     (4 fns: `is_svg_string`,
     `cache_path_for_content`, `write_to_cache`,
     `transform_value`).
   - `is_svg_string(s)` returns true if the
     string starts (after optional whitespace)
     with `<svg` or `<?xml` followed by `<svg`.
     Other formats deferred -- start with SVG;
     PNG / HTML / etc. land later if needed.
   - `cache_path_for_content(content)` produces
     `$MLPL_CACHE_DIR/<sha256-prefix-12chars>.
     svg`. Default cache dir
     `dirs::cache_dir().join("mlpl")` (add
     `dirs = "5"` to `mlpl-cli` Cargo.toml);
     respect `MLPL_CACHE_DIR` env-var override.
     Hash-prefix gives content-addressed naming
     so repeated identical viz calls don't
     create duplicates.
   - `write_to_cache(content) -> PathBuf` --
     creates the cache dir if missing, writes the
     content, returns the path.
   - `transform_value(s) -> String` --
     orchestrator. If `is_svg_string(s)` returns
     true, write to cache and return
     `format!("viz: {}", path.display())`;
     otherwise return `s` unchanged.

2. **Wire it in**.
   - `mlpl-repl`'s value-display path (both local
     and connect mode) routes `Value::Str`
     through `viz_cache::transform_value` before
     printing. Look at where the local REPL
     prints `Value::Str` today (probably in
     `repl.rs`'s output path) and add the
     transform there.
   - Connect mode: the client gets back a JSON
     `value` from the server. After deserializing
     it, run the same transform.
   - **Server side**: do NOT transform on the
     server. The server returns the raw SVG
     string; clients decide locally where to put
     it. (A future saga can add a server-side
     viz storage URL endpoint; out of scope
     here.)

3. **Tests**.
   - `crates/mlpl-cli/tests/viz_cache_tests.rs`:
     SVG detection (positive: `<svg ...>`,
     `  <svg>`, `<?xml ...?><svg>`; negative:
     `"hello"`, `"<div>not svg</div>"`).
     `cache_path_for_content` is deterministic
     (same input -> same path) and
     content-addressed (different input ->
     different path). `write_to_cache` writes
     content correctly and returns a readable
     path; use `tempfile::TempDir` + override
     `MLPL_CACHE_DIR` for isolation.
     `transform_value` returns the path string
     for SVG and passes through non-SVG.

4. **`docs/using-cli-server.md`** retrospective +
   user guide. Sections:
   - Status block (shipped Saga 21 / v0.17.0).
   - What this is about (multi-client MLPL via a
     long-running interpreter).
   - `mlpl-serve` quickstart: `cargo run -p
     mlpl-serve -- --bind 127.0.0.1:6464 --auth
     required`; copy the printed startup token
     for the first session creation OR show the
     two-step "create session, then eval" curl
     dance.
   - `mlpl-repl --connect <url>` walkthrough.
     Cover: it creates a session for you, line-
     by-line evals run server-side, slash
     commands fetch JSON snapshots, `:ask` still
     works locally.
   - The CLI viz cache strategy + the
     `MLPL_CACHE_DIR` env var.
   - Security posture: constant-time token
     compare, `--bind 0.0.0.0` requires
     `--auth required`, no LLM proxy yet (browser
     `llm_call` still blocked).
   - The multi-client picture today (CLI client
     works; web UI stays WASM in MVP; ratatui /
     Emacs / desktop GUI all future sagas).
   - Non-goals deferred: server-side LLM proxy,
     visualization storage URLs, SSE streaming,
     cancellation, web UI re-routing, persistence.

5. **`docs/configurations.md`** refresh.
   - The CLI server column has cells today
     (`yes (proxy)`, `mem only`, etc.) that anticipated
     this saga. Update them: cells that are now
     actually implemented stay `yes`; cells that
     remain post-MVP get a footnote pointing at
     `using-cli-server.md`'s "Non-goals" section.
   - Update footnote [3] (CORS / proxy story)
     and footnote [7] (`llm_call` web access) to
     reflect that the CLI server skeleton is now
     shipped but the proxy specifically is still
     deferred.

6. Quality gates + commit. Commit message
   references Saga 21 step 003.
