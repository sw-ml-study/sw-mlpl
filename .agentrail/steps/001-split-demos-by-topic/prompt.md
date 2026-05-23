Saga 33 step 001: split demos.rs by topic.

apps/mlpl-web/src/demos.rs is 1179 lines with 29 inline Demo struct literals. Pure compile-time data conflated into one file. Per docs/loose-coupling.md phase 1: pull each topical group into a sibling file as named pub const declarations.

Plan:
1. Create 5-6 new sibling files in apps/mlpl-web/src/ alongside demos.rs:
   - demos_basics.rs (arithmetic / arrays / broadcasting / reduce: ~8 entries)
   - demos_models.rs (linear / chain / residual / attention / embed / lora: ~6 entries)
   - demos_training.rs (train loops / loss curves / autograd walk-throughs: ~5 entries)
   - demos_viz.rs (histogram / scatter / decision boundary / embedding viz: ~4 entries)
   - demos_advanced.rs (tiny LM / ViT / multi-head attention: ~3 entries)
   - demos_mlx.rs (MLX-routed variants: ~3 entries)
2. Each Demo {...} literal in demos.rs becomes 'pub const NAME_IN_SCREAMING_CASE: Demo = Demo { ... };' in the appropriate topical file.
3. demos.rs keeps the Demo struct, the ProgressNote machinery + PROGRESS_NOTES table + progress_notes_for() fn, and rewrites 'pub const DEMOS: &[Demo] = &[crate::demos_basics::ANALYSIS_HELPERS, crate::demos_basics::BASIC_ARRAYS, ...]' as a thin facade.
4. Register the new modules in main.rs (or wherever lib.rs lives for mlpl-web).

Target: demos.rs 1179 -> ~150-200 lines (retires File-LOC FAIL). Each topical sibling 150-250 lines (PASS). No new FAILs introduced -- apps/mlpl-web Crate-Module-Count is already FAILing so adding modules is free for that metric.

Strict gate: sw-checklist net-negative on BOTH fails AND warnings vs HEAD~1. Workspace tests + clippy + fmt + markdown-checker green.