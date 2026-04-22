Phase 3 step 006: web tutorial lesson + `docs/using-perturbation.md`.

User-facing retrospective and live-demo surface for the
Neural Thickets story.

1. Add a "Neural Thickets" tutorial lesson to the web REPL
   (`apps/mlpl-web/src/tutorial.rs`) matching the Saga 14
   "Running on MLX" pattern:
   - Short narrative explaining the four builtins.
   - The demo source (can be a cut-down of
     `demos/neural_thicket.mlpl` so it renders a heatmap
     quickly in the browser).
   - Rendered heatmap via `svg(..., "heatmap")`.
   - If CPU run time is too slow to be interactive in the
     browser, keep the fully-fledged demo in
     `demos/neural_thicket.mlpl` and ship a smaller variant
     in the tutorial (explicit in the lesson text).
2. Web UI change => rebuild `pages/` with
   `./scripts/build-pages.sh` and commit both sources and
   `pages/` together per CLAUDE.md (live demo ships from the
   committed `pages/` dir; project memory:
   `feedback_pages_rebuild.md`).
3. Write `docs/using-perturbation.md` covering:
   - The four builtins, with one-paragraph rationale each.
   - The four families and which params each touches.
   - The measured heatmap pattern from
     `demos/neural_thicket.mlpl` (a screenshot or an inline
     SVG snippet is ideal -- whichever matches existing docs
     style).
   - Honest MLX vs. CPU numbers from step 005.
   - Follow-up surface we did NOT ship: depth-aware families
     (`early_N_layers` / `late_N_layers`), low-rank
     perturbation (`perturb_low_rank`), real pretrained
     checkpoints (Saga 15+), LLM sidecar (Saga 19).
4. `markdown-checker -f "**/*.md"` on the new / edited docs
   (ASCII-only reminder -- no smart quotes, no em-dashes).
5. Quality gates + `/mw-cp`. Commit message references
   Saga 20 step 006.
