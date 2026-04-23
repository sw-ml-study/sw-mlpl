Phase 3 step 007: release v0.13.0.

Cut the LoRA release.

1. Bump `Cargo.toml` `workspace.package.version` from
   `0.12.0` to `0.13.0`. All workspace members inherit via
   `version.workspace = true`; no per-crate edits.

2. Update `CHANGELOG.md` with a v0.13.0 section above the
   v0.12.0 entry:
   - New builtins: `freeze`, `unfreeze` (if shipped),
     `lora`.
   - New demos: `demos/lora_finetune.mlpl`,
     `demos/lora_finetune_mlx.mlpl`.
   - New docs: `docs/using-lora.md` + tutorial lesson.
   - Measured MLX-vs-CPU numbers for the LoRA training
     step.
   - Scope notes: what shipped vs what is deferred
     (QLoRA, selective attachment, merge_lora).

3. Mark Saga 15 complete in `docs/saga.md` with a new
   retrospective entry above the Saga 20 entry (newest-
   first), and update `docs/status.md`:
   - Move the Saga 15 row from Planned to Completed.
   - Roll the remaining Planned sagas' target versions
     forward by one minor as necessary.

4. `cargo build --release` to confirm the bump compiles.
   Do NOT run `sw-install` (project memory:
   `feedback_no_sw_install.md`).

5. `/mw-cp` quality gates: cargo test, clippy, fmt,
   markdown-checker on the updated docs, sw-checklist.

6. Tag `v0.13.0` locally. Do NOT push the tag unless the
   user confirms (matching the v0.12.0 cadence).

7. `agentrail complete --done` closes the saga.
