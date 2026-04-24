Phase 3 step 003: release v0.16.0.

1. **Version bump**. `Cargo.toml`
   workspace.package.version `0.15.0 -> 0.16.0`.
   Minor-level bump because the surface adds a new
   language-level builtin (not a patch).

2. **`CHANGELOG.md`** new v0.16.0 section above
   v0.15.0 (Saga 22's release):
   - Added: `llm_call(url, prompt, model) -> string`
     builtin (with contract link).
   - Added: `demos/llm_tool.mlpl` (CPU, CLI-only).
   - Added: `docs/using-llm-tool.md` retrospective.
   - Changed: `:ask` REPL command now uses the
     language-level `llm_call` under the hood;
     system + user context framing preserved but
     POSTed through the same HTTP path as `.mlpl`
     scripts.
   - Tests: list counts for `llm_call_tests.rs`.
   - Scope notes: streaming, tools, chat threading,
     batching, teacher distillation, web/WASM
     support all remain deferred.

3. **`docs/saga.md`**. Saga 19 retrospective entry.
   Place it as a new top-level section above
   Saga 22 (the most recent retrospective) to
   preserve reverse-chronological ordering.

4. **`docs/status.md`**. Saga 19 row moves from
   Planned to Completed with `[x]` and the v0.16.0
   version tag. Update the "Next saga to start"
   pointer to Saga 21 (CLI server) — it is next
   before the Linux move because once dev goes off
   Apple Silicon the browser live demo loses local
   MLX, and the server-side `mlpl-serve` route
   keeps it usable.

5. `cargo build --release` to confirm the bump.

6. `./scripts/build-pages.sh` to rebuild pages/
   (docs/configurations.md + new lesson intro
   may have changed); commit the new WASM bundle.

7. `./scripts/gen-changes.sh` to refresh CHANGES.md;
   commit.

8. `/mw-cp` quality gates (cargo test, clippy,
   fmt, markdown-checker on changed files, sw-
   checklist baseline hold).

9. Tag `v0.16.0` locally. DO NOT push the tag
   without explicit user confirmation (v0.13.0 /
   v0.14.0 / v0.14.1 / v0.15.0 cadence).

10. `agentrail complete --done`.
