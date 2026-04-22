Phase 2 step 004: `demos/neural_thicket.mlpl` end-to-end on CPU.

First runnable Neural Thickets demo, CPU-only. No
`device("mlx")` block yet -- that is step 005.

1. Author `demos/neural_thicket.mlpl` closely mirroring the
   strawman in `docs/mlpl-for-neural-thickets.md`, but:
   - Use the `load_preloaded("tiny_shakespeare_snippet")`
     corpus (Saga 12 surface).
   - Keep the base training short enough to run interactively
     on CPU (e.g. `train 200` or fewer, document the choice in
     a comment -- the demo argues for perturbation, not base
     quality).
   - Sweep 4 families x 4 seeds = 16 variants.
   - Score each on held-out validation tokens ("to be or not
     to be that is the question" per the design sketch, or a
     similar short held-out string if that one is too short
     to form 16 pairs at the chosen context).
   - Build `losses : [16]` via a `repeat 16 { scatter(...)
     }` loop.
   - `best_idx = argtop_k(neg(losses), 4)`.
   - Ensemble loop: iterate `best_idx`, rebuild each variant
     via `clone_model` + `perturb_params` with the matching
     seed, average their logits.
   - `heat = reshape(losses, [4, 4])`, `svg(heat, "heatmap")`.
2. Run with `mlpl-repl -f demos/neural_thicket.mlpl` (never
   pipe on stdin -- `repeat` + stdin splitting is a known
   footgun, see project memory on `mlpl_repl_script_mode`).
3. Integration test in
   `crates/mlpl-eval/tests/neural_thicket_tests.rs` that
   runs a cut-down version (smaller V, smaller context,
   fewer training steps, still 4 x 4 variants) and asserts:
   - The heatmap result has shape `[4, 4]`.
   - All 16 loss entries are finite.
   - `argtop_k` returns 4 indices all in `[0, 16)`.
   - Ensemble logits have the same shape as a single variant's.
4. Add a one-line entry to `docs/demos-scripts.md` pointing
   at the new demo.
5. Quality gates + `/mw-cp`. Commit demo, test, and doc
   entry together. Commit message references Saga 20 step
   004.
