Phase 3 step 007: release v0.12.0.

Cut the Neural Thickets release.

1. Bump workspace version in top-level `Cargo.toml` from
   the current v0.11.x to `0.12.0`. Update every workspace
   member that pins an internal `mlpl-*` dependency to the
   new version.
2. Update `CHANGELOG.md` with a v0.12.0 section listing:
   - New builtins: `clone_model`, `perturb_params`,
     `argtop_k`, `scatter`.
   - New demo `demos/neural_thicket.mlpl`.
   - New docs: `docs/using-perturbation.md` + tutorial lesson.
   - Measured MLX-vs-CPU numbers for the variant loop.
3. Mark Saga 20 complete in `docs/saga.md` (or the
   equivalent saga-index doc); keep the entry pointing at
   `docs/mlpl-for-neural-thickets.md` and
   `docs/using-perturbation.md`.
4. `cargo build --release` to confirm the version bump
   compiles cleanly. Do NOT run `sw-install` -- that
   overwrites the user's stable installed mlpl-repl
   (project memory: `feedback_no_sw_install.md`); only run
   it if the user explicitly asks.
5. Tag the release commit locally with `v0.12.0` (do not
   push the tag unless the user asks -- confirm before
   publishing).
6. Quality gates + `/mw-cp` for the release commit.
7. `agentrail complete --done` closes the saga.
