Carve demos cluster out of mlpl-web into components/web-demos/ workspace. 12 source files: demos.rs (Demo+ProgressNote+PROGRESS_NOTES+DEMOS+progress_notes_for+3 inline tests) + 11 demos_*.rs files (pure data, only 'use crate::demos::Demo'). 13 modules > 7 limit -> must split into 4 sub-crates within the component (per the <=4-crates-per-workspace HARD RULE).

Sub-component architecture:
  mlpl-web-demos-types   -> lib.rs facade + registry.rs (Demo, ProgressNote, PROGRESS_NOTES, progress_notes_for) [2 modules]
  mlpl-web-demos-basic   -> lib.rs facade + basics, lm, udf [4 modules]
  mlpl-web-demos-vision  -> lib.rs facade + attention, vit, cnn [4 modules]
  mlpl-web-demos         -> lib.rs facade + aggregator.rs (DEMOS const) + autoencoder, gan, models, dim_reduction, rnn [7 modules]

Dep graph: -basic / -vision / facade each depend on -types. Facade additionally depends on -basic + -vision. DAG. <=4 crates per component.

mlpl-web wires: 'pub use mlpl_web_demos as demos;'. The 5 callers (component_mode_bar, paths_view, readme_counts, handlers_demo, mode_path) keep using crate::demos::DEMOS / Demo / progress_notes_for unchanged.

mlpl-web modules 73 -> 61. The 3 inline tests move to tests/registry.rs in -types so the test fn count stays out of any src module count.