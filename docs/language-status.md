# Language status dashboard

**One-screen view of where MLPL's language fixes stand.** The
catalog of findings lives in `docs/language-audit.md`; the queue
lives in `docs/plan.md`; this doc says what is *happening right
now* and what has *shipped*.

Update this doc whenever a saga step lands that changes the
status of any finding. Saga 30's step 006 doubles as the
audit-closeout step; analogous steps in later sagas should do the
same for their findings.

Last refreshed: 2026-05-27 (saga 55 closed; runtime component).

## Active saga

None. Saga 55 (`component-runtime`) closed 2026-05-27.

`agentrail status` is the live source of truth; this row is the
human-readable summary.

## Saga timeline (oldest first)

| Saga                  | Status     | Audit findings touched | Notes                                       |
|-----------------------|------------|------------------------|---------------------------------------------|
| `vit` (29)            | shipped    | -- (capability saga)   | Closed 2026-05-20. Archived under `.agentrail-archive/`. |
| `tier1-cleanup` (30)  | shipped    | #18, #19               | Closed 2026-05-20. Six steps; both findings retired. |
| `scripting-cluster` (31) | shipped | #22, #23, #24, #25, #26, #27, #28, #29, #30 | Closed 2026-05-22. Eight steps; turned MLPL into a real scripting language (if/else, while/break/continue, args + CLI passthrough + positional script path, to_number / to_int / env, print / eprint, read_stdin / read_stdin_lines, exit + Err-as-exit-1, `#!/usr/bin/env mlpl-repl` shebang support, `demos/classify.mlpl` worked example). |
| `tech-debt-paydown` (32) | shipped | -- (process saga) | Closed 2026-05-22. Eight steps; **delivered -8 fails / -2 warnings vs the "halve both" target of -76 fails / -227 warnings**. Six new sibling crates extracted as clean DAG leaves (mlpl-eval-core, mlpl-runtime-core/-data/-dim-reduction, mlpl-web-eval/-lessons/-path-body) plus several in-crate splits (experiment_compare, inspect_render, image_io_pixels, model_tape_attention, fetch_io, dataset_helpers, ops_concat, gallery_layout). Step 007 documented the "compose-don't-compress" lesson; step 008 wrote `docs/loose-coupling.md` as the canonical HOW-TO. |
| `env-demos-paydown` (33) | shipped | -- (mixed saga) | Closed 2026-05-25. 48 steps. env.rs split (55->3 fns), demos.rs split (1179->150 lines), model_dispatch.rs split, DR milestone (UMAP, MDS, random projection, PCA loadings/variance, knn_graph, critical_dimensions viz, 6 lessons, 5 demos, 1 learning path), REPL completion popup (Ctrl+Space trigger, arrow-key navigation), visual regression harness, moons MLP fix, perplexity builtin. sw-checklist 143->130 fails, 450->460 warnings. |
| `splash-tour` (34) | shipped | -- (UX saga) | Closed 2026-05-25. 6 steps. Splash overlay (first-visit welcome with splash-bg.png, 4 quick-start cards), 6-step guided tour (CSS spotlight, tooltip positioning), what's-new modal (version-bump triggered), Tour header button, Escape closes overlays. Decomposed components.rs (6 files) + handlers.rs (3 files) by concern. Process docs: feature-design-process.md, sw-checklist-mitigations.md, mlpl-web-architecture.md. |
| `3d-viz-stage` (35) | shipped | -- (capability saga) | Closed 2026-05-25. 7 steps. 3D visualization stage: :3d toggle + Ctrl+3, Three.js scene with OrbitControls, step event pipeline (eval emits shapes), shape-aware sculptures (scalar/vector/matrix/tensor), demo integration, click-to-inspect (:describe via raycaster), nav buttons + arrow keys, fog. |
| `script-editor` (36) | shipped | -- (UX saga) | Closed 2026-05-26. 4 steps. Script editor tab with Run/Load/Save/Clear, file picker for .mlpl upload, Ctrl+Enter to run, Blob download. |
| `3d-scale-connections` (37) | shipped | -- (capability saga) | Closed 2026-05-26. 4 steps. Log-proportional sizing, scale legend, connection arrows between dependent steps, :3d reset command. |
| `element-data-pipeline` (38) | shipped | -- (capability saga) | Closed 2026-05-26. 5 steps. eval_with_values API, element values in Stage3dEvent, value-colored sculptures (bar charts, cell grids, diverging colormap), detail panel with stats + histogram. |
| `cnn-builtins-viz` (39) | shipped | -- (capability saga) | Closed 2026-05-27. 6 steps. conv2d, pool2d, relu standalone, Simple CNN demo, 3D stacked heatmap channels for rank-4 tensors. |
| `autoencoder-demo` (40) | shipped | -- (content saga) | Closed 2026-05-27. 4 steps. Autoencoder demo (encoder 8->3, decoder 3->8), bottleneck hint in 3D detail, glossary entries (Bottleneck, Reconstruction Error), salt flat backdrop. |
| `rnn-lstm-builtins` (41) | shipped | -- (capability saga) | Closed 2026-05-27. 5 steps. rnn_cell and lstm_cell builtins, RNN sequence demo (5-step), LSTM sequence memory demo (10-step), glossary (RNN, LSTM, Vanishing Gradient, Hidden State). |
| `gan-demo` (42) | shipped | -- (content saga) | Closed 2026-05-27. 3 steps. sin/cos builtins, GAN (2D circle) demo with alternating adam, glossary (GAN, Generator, Discriminator, Adversarial Training), BUILTIN_GROUPS completeness fix (11 builtins). |
| `architecture-zoo-path` (43) | shipped | -- (content saga) | Closed 2026-05-27. 2 steps. "From Pixels to Language" learning path with 4 groups (See/Remember/Create/Attend), 27 steps. Updated 3 stale Visual path diagram entries. |
| `chronological-history-path` (44) | shipped | -- (content saga) | Closed 2026-05-27. 2 steps. "A chronological history of ML" path with 24 eras (1943-2023), year-centric framing. |
| `data-exploration-path` (45) | shipped | -- (content saga) | Closed 2026-05-27. 3 steps. Upload & Inspect Image demo (Basics), Data & Exploration path (16 steps), REPL-to-Script milestone doc. |
| `udf-control-flow` (46) | shipped | -- (capability saga) | Closed 2026-05-27. 5 steps. def u:name(args) { body }, return, pi(), e(), range() alias for iota(), UDF demo, 16 eval tests, glossary (UDF, Scope, Recursion). |
| `repl-to-script-path` (47) | shipped | -- (content saga) | Closed 2026-05-27. 2 steps. "REPL to Script" learning path (9 steps). |
| `training-paradigms-path` (48) | shipped | -- (content saga) | Closed 2026-05-27. 2 steps. "Training Paradigms" learning path (16 steps, 4 groups). |
| `optimizers-regularization-path` (49) | shipped | -- (content saga) | Closed 2026-05-27. 2 steps. "Optimizers & Regularization" learning path (19 steps). |
| `warning-ratchet-spike` (50) | shipped | -- (refactor saga) | Closed 2026-05-27. 5 steps. Extracted 6 new crates from mlpl-runtime + mlpl-parser: mlpl-runtime-math, mlpl-runtime-conv, mlpl-runtime-rnn, mlpl-runtime-array, mlpl-runtime-ml, mlpl-lexer. Plus ast_fmt_compound.rs helper in parser. sw-checklist 141->133 fails (-8), 473->474 warnings. Big wins: builtins.rs 16->3 fns, mlpl-runtime 9->4 modules, mlpl-parser 10->6 modules, conv2d/pool2d/lstm_cell/fmt all retired (FAIL->WARN or PASS). Note: extractions landed flat in `crates/`; component-restructure saga (51+) will migrate them into `components/`. |
| `shared-target-infra` (51) | shipped | -- (infra saga) | Closed 2026-05-27. 2 steps. Added `.cargo/config.toml` at repo root with `[build] target-dir = "target"` so every workspace (main, components/*, services/*) writes to ONE shared target/ tree. Reclaimed ~2GB of stale per-component target dirs. Prerequisite for the component-migration sagas (52+) that will migrate the workspace from flat `crates/` into `components/<feature>/crates/` bottom-up. |
| `component-lang-core` (52) | shipped | -- (structural saga) | Closed 2026-05-27. 5 steps. First component-migration saga: created `components/lang-core/` nested workspace and moved the three foundational crates (mlpl-core, mlpl-array, mlpl-eval-core) into it. Updated 51 dependent Cargo.toml path references across all four workspaces (main, lang-core, mlpl-session, mlpl-mlx-serve). sw-checklist: 221->222 passed (+1), 133 fails unchanged, 474 warnings unchanged. Structural -- enables future grouped extractions inside lang-core (e.g., splitting mlpl-array's 13 modules across sibling crates). Pattern established for subsequent sagas 53+: one saga per component, bottom-up. |
| `lang-core-decompose-array` (53) | shipped | -- (refactor saga) | Closed 2026-05-27. 7 steps. Decomposed mlpl-array (13 modules, Crate Module Count FAIL) into 5 sparse sibling crates within components/lang-core/ using the extension-trait pattern: `mlpl-array-ops-matmul` (MatmulExt, DotExt), `-reduce` (ReduceAxisExt, ArgmaxAxisExt), `-compose` (ConcatExt, StackExt fn, PatchifyExt, TakeExt), `-shape` (ReshapeExt, TransposeExt), `-element` (ApplyBinopExt). Call sites preserved via `use mlpl_array_ops_*::prelude::*;` -- no API churn for the 600+ `a.matmul(&b)`-style calls. sw-checklist: 222->247 passed (+25), **133->132 fails (-1 FAIL retired)**, 474->472 warnings (-2). Each op's body opportunistically shrunk while migrating (matmul 50->32, reduce_axis 46->11, argmax_axis 44->13, etc.). Establishes the move-AND-split pattern: component migrations must split crowded crates inside the component, not just move them. |
| `component-lang-syntax` (54) | shipped | -- (structural saga) | Closed 2026-05-27. 6 steps. Created components/lang-syntax/ with the source-text-to-AST family. Moved + decomposed mlpl-lexer into 7 sparse siblings (mlpl-lexer-token, mlpl-lexer-error, mlpl-lex-string, mlpl-lex-number, mlpl-lex-punct, mlpl-lex-ident, mlpl-lexer orchestrator). Moved + split mlpl-parser into mlpl-parser-ast (types + Display impls, Display orphan rule) and mlpl-parser (Parser logic + re-exports for downstream backward compat). Moved mlpl-macro and mlpl-lower-rs unchanged. sw-checklist: 247->287 passed (+40), 132 fails unchanged, 472 warnings unchanged. Big PASS gain from 9 new sparse crates each contributing clean PASS lines. mlpl-lexer's prior lex_util 6-fn WARN + lex_ident 32-LOC WARN retired through structural split. |
| `component-runtime` (55) | shipped | -- (structural saga) | Closed 2026-05-27. 4 steps. Created components/runtime/ and bulk-moved 11 runtime crates from crates/: mlpl-runtime (dispatch), mlpl-runtime-core, and the 9 concern-grouped sibling crates (math, conv, rnn, array, ml, data, dim-reduction, umap, mds-rp -- already sparse from saga 50). Updated 5 external consumer Cargo.toml refs and fixed intra-component cross-references to lang-core. sw-checklist: 287->288 passed (+1), 132 fails unchanged, 472 warnings unchanged. The runtime family had no remaining FAILs (saga 50 already retired them). Several WARNs remain (dim-reduction 7 modules, validate fns over 25 LOC); flagged for a future `runtime-warn-paydown` saga. |

The "proposed" sagas have full milestone docs; the user has
confirmed the editorial stances. They are not yet initialized in
agentrail.

## Per-finding status (critical tier)

| #   | Short title                                  | Status      | Owning saga / step       | Shipped commit |
|-----|----------------------------------------------|-------------|--------------------------|----------------|
| #1  | Closures don't differentiate                 | proposed    | future autograd-lift saga | --             |
| #2  | `device("mlx")` param relocation             | proposed    | future                   | --             |
| #3  | Booleans encoded as `0.0` / `1.0` floats     | proposed    | future                   | --             |
| #10 | No `vmap` / batched transform                | proposed    | future                   | --             |
| #12 | No `gather` / no slice ranges                | proposed    | future                   | --             |
| #15 | Inline forward expression anti-pattern       | downstream of #1 | --                  | --             |
| #18 | `concat` axis restricted to `{0, 1}`         | **shipped** | saga 30 steps 001 (forward) + 002 (backward) | 001: `c133d57`, 002: `4e27f9c` |
| #19 | Multi-head attention has forward-only tape   | **shipped (stale audit)** | originally saga 29 step 013; verified in saga 30 step 004 | `66d63c9` |
| #22 | No `if` / `else`                             | **shipped** | saga 31 step 004         | `29f6d3a`      |
| #24 | No CLI argument capture in script mode       | **shipped** | saga 31 step 003         | `cbba20a`      |
| #26 | No string-to-number parsing                  | **shipped** | saga 31 step 002         | `87f4a2b`      |
| #28 | No `print` / explicit script output          | **shipped** | saga 31 step 001         | `4f7f1f2`      |

## Per-finding status (nice-to-have)

| #   | Short title                                  | Status   | Owning saga       |
|-----|----------------------------------------------|----------|-------------------|
| #4  | Magic seed constants                         | proposed | future            |
| #5  | `:upload` stringly-typed error kinds         | proposed | future            |
| #6  | `concat` arity overload / list-variadic      | proposed | (could rider on saga 30) |
| #8  | Stringly-typed `svg()` viz type names        | proposed | future            |
| #9  | Inconsistent axis position across builtins   | proposed | future            |
| #11 | No `jit` / compilation boundary              | proposed | future            |
| #14 | No named-axis types                          | proposed | saga 19 (queued)  |
| #16 | Model-DSL doesn't cover `take` / `reshape`   | proposed | future            |
| #17 | Stringly-typed device names                  | proposed | future            |
| #23 | No `while` / `break` / `continue`            | **shipped** (saga 31 step 005, `5509e72`) | scripting saga    |
| #25 | No `env()`                                   | **shipped** (saga 31 step 002, `87f4a2b`) | scripting saga    |
| #27 | No stdin read                                | **shipped** (saga 31 step 006, `24f1a31`) | scripting saga    |
| #29 | No script exit code                          | **shipped** (saga 31 step 006, `24f1a31`) | scripting saga    |
| #30 | No script-mode example demo                  | **shipped** (saga 31 step 007, `4a67ae8`) | scripting saga    |

## Per-finding status (cosmetic)

| #   | Short title                                  | Status   |
|-----|----------------------------------------------|----------|
| #7  | Inconsistent builtin naming convention       | proposed |
| #13 | No tacit / point-free programming            | deferred (per audit) |
| #20 | `BUILTINS` slice has implicit ordering       | proposed |
| #21 | sw-checklist budget shapes the code          | process, out of scope for audit |

## Shipped (most recent first)

- **2026-05-25** -- saga 33 (`env-demos-paydown`) closed.
  48 steps spanning structural paydown + the dimensionality
  reduction capability milestone + REPL UX improvements.
  **Dimensionality reduction milestone** (steps 030-048):
  new builtins `pca_components(X, k)`,
  `pca_variance_explained(X, k)`, `knn_graph(X, k)`,
  `umap(X, n_neighbors, min_dist, iters, seed)`,
  `mds(X, k, iters, seed)`,
  `random_projection(X, k, seed)`; new viz type
  `critical_dimensions`; 5 demos (PCA 3D interactive, PCA
  loadings, UMAP vs PCA, UMAP vs t-SNE, Dim-reduction
  zoo); 6 tutorial lessons (Why reduce dimensions, PCA
  the linear baseline, SNE the very-slow ancestor, t-SNE
  a peek at nonlinear methods, UMAP the modern default,
  Reading a critical-dimensions heatmap); 1 learning path
  ("Dimensionality reduction"). New crates:
  `mlpl-runtime-umap`, `mlpl-runtime-mds-rp`. **REPL
  completion** (steps 044-047): Ctrl+Space popup with
  candidate matching, arrow-key navigation, Enter/Escape/
  ArrowRight acceptance. **Other**: visual regression test
  harness, moons MLP fix, `perplexity()` builtin, scatter
  legend outside plot, path-resume bug fix.
- **2026-05-22** -- saga 32 (`tech-debt-paydown`) closed.
  Eight steps; sw-checklist 151 -> 143 fails (-8), 452 ->
  450 warnings (-2). Far short of the "halve both" target
  (~75 / ~225). Six new sibling crates extracted as
  strict-DAG leaves:
  - `mlpl-eval-core` (model + metric_sink + inspect_groups
    leaf types).
  - `mlpl-runtime-core` (RuntimeError + Xorshift64 leaf
    types) +  `mlpl-runtime-data` (dataset_builtins +
    embedding_builtins + grid_builtin) + `mlpl-runtime-dim-
    reduction` (t-SNE + PCA).
  - `mlpl-web-eval` (eval pipeline + state + summary),
    `mlpl-web-lessons` (static tutorial content),
    `mlpl-web-path-body` (markdown-ish renderer).
  Plus in-crate splits: `experiment_compare.rs`,
  `inspect_render.rs`, `image_io_pixels.rs`,
  `model_tape_attention.rs`, `fetch_io.rs`,
  `dataset_helpers.rs`, `ops_concat.rs`, `gallery_layout.rs`.
  Step 007 was a no-progress step that surfaced the
  "compose-don't-compress" anti-pattern: in-place
  compressions of 27-line warning functions fight rustfmt
  and net zero. Step 008 captured the lesson in
  `docs/loose-coupling.md` (the canonical HOW-TO doc for
  loose coupling refactors) and updated CLAUDE.md +
  `docs/code_metrics.md` + memory entries to point at it.
  Highest-value carry-over for the next tech-debt saga:
  apply the phase-separation lens (compile-time / start-up
  / conditional / dataflow) to `env.rs` (55 fns),
  `demos.rs` (1179 lines), and `model_dispatch.rs`
  (905 lines + the 100-LOC `apply_model` FAIL). All three
  conflate phases; splits by phase will retire 5+ FAILs
  per file.

- **2026-05-22** -- saga 31 closed (`scripting-cluster`).
  Eight steps shipped audit findings #22 / #24 / #25 / #26 /
  #28 (the four critical scripting findings) plus #23 / #27 /
  #29 / #30 (the nice-to-have block). MLPL now functions as a
  real scripting language: take CLI args via `args()`, branch
  on flags via `if / else`, loop via `while / break /
  continue`, read stdin, parse strings via `to_number /
  to_int / env`, print via `print / eprint`, control exit via
  `exit(code)` + automatic `Err(...)` -> exit-1, and run as
  `#!/usr/bin/env mlpl-repl` shebang executables. Worked
  example in `demos/classify.mlpl`. See the saga timeline
  table above for the per-step commit list.
- **2026-05-22** -- saga 31 step 007: `demos/classify.mlpl`
  + positional script path shipped (commit `4a67ae8`).
  Score-classifier worked example exercises every saga 31
  builtin / form in ~50 lines; positional `mlpl-repl
  script.mlpl` enables `#!/usr/bin/env mlpl-repl` shebang
  scripts. 7 demo subprocess tests + 8 positional-path
  tests. Closes audit #30. sw-checklist held.
- **2026-05-22** -- saga 31 step 006: stdin / exit / Err
  propagation shipped (commit `24f1a31`). `read_stdin()` /
  `read_stdin_lines()` (block until EOF, TTY-refuse via
  `IsTerminal`); `exit(code)` (validated 0..=255 then
  `std::process::exit`); `mlpl-repl -f` mode now maps a
  final `Err(msg)` to exit 1 + stderr. 11 subprocess tests +
  8 in-process arity / range tests. Refactor: extracted
  `eval_program.rs` and `script_mode.rs` retired 2
  pre-existing FAILs. Closes audit findings #27 and #29.
- **2026-05-22** -- saga 31 step 005: `while` / `break` /
  `continue` shipped (commit `5509e72`). `while cond {
  body }` re-evaluates until cond is falsy or break; `break`
  / `break value` exits the nearest enclosing while; bare
  `break` yields 0. `continue` skips to the next condition
  check. Break/continue outside a loop is a runtime error
  (`LoopControlOutsideLoop`). 15 tests in
  `while_loop_tests.rs`. Refactor: extracted
  `crates/mlpl-parser/src/ast_fmt.rs` retired ast.rs 8-fn
  FAIL. Closes audit #23.
- **2026-05-21** -- saga 31 step 004: `if cond { then } else
  { else }` expression shipped (commit `29f6d3a`). First surface
  conditional in MLPL; an EXPRESSION (returns a value), not a
  statement. Truthy on non-zero scalars and `Ok(_)` Results;
  falsy on `0.0` and `Err(_)`. The `else` clause is required;
  both branches can return any Value type. 14 new tests in
  if_else_tests.rs. New TokenKind::If / ::Else; new Expr::If AST
  variant; new parser rule; early-return eval intercept in the
  Device-block style. Closes audit #22. sw-checklist
  ratchet-down: -1 fail, -1 warning. The bigger ratchet-down
  refactor (splitting fat crates / files) is a separate
  upcoming commit -- see CLAUDE.md (now updated to clarify
  the ratchet rule is REDUCE, not HOLD).
- **2026-05-21** -- saga 31 step 003: args() builtin + CLI
  passthrough shipped. Two-part change: args() returns a StrList
  of the trailing CLI args (after `--`) in mlpl-repl -f mode;
  list_get(xs, i) -> Result added because StrList had no
  per-element accessor (you can't index args() without it).
  Environment carries the args via a new pub(crate) field +
  set_cli_args() setter. 10 eval-side tests + 4 binary-spawn
  integration tests in apps/mlpl-repl/tests/. Closes audit #24.
  Also ratcheted sw-checklist warnings 459 -> 457 by tightening
  patchify_backward, stack_backward (mlpl-autograd) and
  probability_invariant (mlpl-eval) below the 25-line warning
  threshold.
- **2026-05-21** -- saga 31 step 002: to_number(s), to_int(s),
  env(name) builtins shipped (commit `87f4a2b`). All three return
  Value::Result so callers branch explicitly on failure via
  is_ok / unwrap_or / err_message. Implementation in
  crates/mlpl-eval/src/result_ops.rs (no new modules because
  mlpl-eval is already at the sw-checklist module-count cap).
  Closes audit findings #25 and #26.
- **2026-05-21** -- saga 31 step 001: print(v) / eprint(v) builtins
  shipped (commit `4f7f1f2`). Eval-side dispatch in
  crates/mlpl-eval/src/eval.rs; writes v's Display form to stdout
  / stderr with newline and returns v unchanged so calls compose
  into expressions. Closes audit #28.
- **2026-05-20** -- saga 30 step 006 (FINAL, saga closed): closed
  out audit findings #18 and #19 in `docs/language-audit.md`
  with shipped headers + commit SHAs; moved both findings into a
  new "Shipped" subsection at the top of `docs/plan.md`'s
  Breaking-change candidates. Saga 30 (`tier1-cleanup`) complete.
- **2026-05-20** -- saga 30 step 005: tightened the multi-head pets
  demo intros/takeaways to state concrete accuracy ("training
  accuracy = 1.0 in 30 adam steps") and dropped a "should look
  DIFFERENT" hedge in the attention-overlay intro to "look
  DIFFERENT after training." The existing strings were already
  describing trained behavior accurately (the demo authors were
  not fooled by the stale audit); these refinements just make
  the empirical claim concrete and verifiable.
- **2026-05-20** -- saga 30 step 004: audit finding #19 was stale.
  Empirical verification: `vit_multihead_quick.mlpl` (heads=4,
  100 adam steps, 20 samples) reaches accuracy 1.0; the browser
  config (8 samples, 30 steps) also reaches loss ~0 and accuracy
  1.0. The multi-head tape was already lowered in saga 29 step
  013 (reshape + take + per-head SDPA + `Tensor::stack`); the
  audit was written from an earlier mid-saga-29 state and never
  refreshed. Added a `multi_head_trains_end_to_end_loss_decreases`
  regression test pinning the behavior. The audit finding has
  been rewritten with a SHIPPED status and a "historical claim
  (now refuted)" section.
- **2026-05-20** -- saga 30 step 003: no live workaround to drop;
  the rank-3 attention path already uses `Tensor::stack` (saga 29
  step 008) which is the correct primitive. Cleaned up two stale
  doc comments (`model_tape.rs` module doc saying "chained concat
  over the head axis"; `Tensor::concat` rustdoc saying "0 or 1
  supported in initial release") that described pre-saga-30
  behavior. Added a `[B=2, T=4, d_model=8]` rank-3 single-head
  regression test pinning the shape and per-batch elementwise
  agreement, so any future regression to a chained-binary-concat
  lowering would fail.
- **2026-05-20** -- saga 30 step 002: audit #18 backward lifted.
  The autograd `concat_backward` now generalizes to any axis,
  matching the forward. Rank-3 and rank-4 finite-difference
  gradchecks pass. Closes the audit finding.
- **2026-05-20** -- saga 30 step 001: audit #18 forward lifted in
  `c133d57`. `mlpl-array::concat` now accepts any `axis` in
  `[0, rank)`.
