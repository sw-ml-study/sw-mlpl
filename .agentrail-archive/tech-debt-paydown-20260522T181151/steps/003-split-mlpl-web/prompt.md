Tech-debt saga step 003: split mlpl-web (29 modules; max 7).

apps/mlpl-web is the binary; the new sibling crates are library crates the binary depends on. Split:

1. apps/mlpl-web (binary, slim shell) -- main.rs + the top-level component glue, ~5-7 modules.

2. crates/mlpl-web-tutorial (new library crate) -- tutorial panel, paths_view, intro_md, lesson rendering. Anything tutorial / learning-path-shaped.

3. crates/mlpl-web-eval (new library crate) -- eval_sse (SSE streaming), upload, viz cache, entry_render, eval-session-result handling. Anything eval-execution-shaped.

Process: same facade pattern. apps/mlpl-web/main.rs's mod declarations get replaced with 'use mlpl_web_tutorial::*' or per-named imports. Yew components keep their existing public types.

Target retirement: 1 Crate-FAIL + ~5-8 Module-Fn-Count FAILs.

Quality gates same as steps 001/002. Bonus: pages/ rebuild after the split (cargo run -p mlpl-web compiles to wasm, ./scripts/build-pages.sh refreshes pages/). Commit pages/ if the bytes differ.