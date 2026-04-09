# Named Axes and Shape Introspection Milestone (Saga 11.5, v0.7.5)

## Why this exists

`docs/are-we-driven-yet.md` flagged named-axes and shape-checking as
the single highest-leverage "agent usability" gap in the
`docs/ai-agent-driven-development.txt` wishlist. It is also the
common pain point behind half the shape errors you hit writing demos
by hand: once an expression has four matmul-reshape-transpose steps,
tracking which axis is which becomes the hard part.

Positional indexing is opaque for both humans and agents:

```mlpl
scores = matmul(Q, transpose(K)) / sqrt(4)   # which axis is which?
```

Named indexing is self-documenting:

```mlpl
Q : [seq, d_k]
K : [seq, d_k]
scores : [seq, seq] = matmul(Q, K.T["seq", "d_k" -> "d_k", "seq"])
```

Saga 11.5 is a *surface-only* milestone: no new kernels, no new
backends, no new ML primitives. It teaches MLPL to carry axis labels
and shape metadata through the evaluator so error messages are
actionable and agents can reason about tensor validity before they
run code.

This saga is inserted between Saga 11 (Model DSL) and Saga 12
(Tokenizers/datasets) because every later saga -- Tiny LM, LoRA,
attention variants, embedding viz -- benefits from labeled axes
and would otherwise accumulate the same hand-written shape-juggling
that the Model DSL just deleted.

## Non-goals

- Full dependent types. Shape labels are metadata, not a type
  system; mismatches are runtime errors with rich context, not
  compile-time rejection. A future saga can tighten this.
- Changes to `DenseArray` row-major storage. Labels live on
  `Value::Array` wrappers, not inside the array buffer.
- New tensor ops. Every existing op keeps its current semantics;
  it just learns to propagate and validate labels.
- Backend work. CPU only.
- Tokenizers, datasets, experiment tracking. That is still Saga 12.

## Quality requirements (every step)

Identical to Saga 11:

1. TDD: failing test first, then implementation, then refactor.
2. Quality gates must all pass before commit:
   - `cargo test`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo fmt --all` + `cargo fmt --all -- --check`
   - `markdown-checker -f "**/*.md"` if docs touched
   - `sw-checklist`
3. Use `/mw-cp` checkpoint process.
4. Push immediately after commit.
5. Web UI changes rebuild `pages/` via `scripts/build-pages.sh`.

## What already exists

- `mlpl_array::Shape` with `dims()` and `rank()` accessors.
- `Value::Array` / `Value::Model` / `Value::Str` enum in
  `mlpl-eval`. Labels would ride on `Value::Array`.
- Structured `EvalError` enum per crate, used everywhere errors
  surface (good substrate for shape-mismatch improvements).
- `:help` REPL command and web help dialog (good surface to
  extend with `:describe <name>`).

## Phases

### Phase 1: Labeled value wrapper

Introduce an axis-labeled shape and thread it through the array
value type without changing any existing behavior.

- New type `LabeledShape { dims: Vec<usize>, labels: Vec<Option<String>> }`
  in `mlpl-array` or `mlpl-core`. `None` labels mean "positional"
  so pre-existing code continues to work unchanged.
- `Value::Array` gains an optional labels field (or wraps in a
  small struct). Default construction leaves labels `None`.
- `shape(x)` continues to return the raw dim vector. New
  `labels(x)` built-in returns a string vector of the axis names
  (empty string for positional axes).
- Zero behavior change until labels are introduced by Phase 2.

### Phase 2: Label introduction primitives

Give users a way to attach labels.

- `label(x, ["batch", "time", "dim"])` tags an existing array.
- Annotation syntax on assignment: `x : [batch, time, dim] = randn(7, [8, 16, 32])`.
  The annotation is optional; omitted means positional.
- `x.T` / `transpose(x)` preserves labels but swaps their order.
- `reshape(x, ...)` clears labels by default (reshape loses
  semantic axis identity) with a `reshape_labeled(x, new_shape, new_labels)`
  opt-in for explicit re-labeling.

### Phase 3: Label-aware ops

Make the existing op set propagate labels through:

- Elementwise (`+`, `-`, `*`, `/`): require matching labels when
  both operands are labeled; union in the usual broadcasting order.
- `matmul(a, b)`: require the contraction axis label to match
  (e.g. `[seq, d] @ [d, heads] -> [seq, heads]`).
- `reduce_add(x, axis)` and friends accept either an integer axis
  or an axis name: `reduce_add(x, "time")`.
- `softmax(x, "d_k")` and `softmax(x, 1)` both work.

At each op, label mismatches raise a structured `EvalError::ShapeMismatch`
with the operator name, expected labels, and actual labels so that
both humans and agents can repair the error without re-running.

### Phase 4: Structured shape errors

Replace the string `EvalError::Unsupported(..)` shape errors with a
dedicated variant:

- `EvalError::ShapeMismatch { op, expected: LabeledShape, actual: LabeledShape }`
  rendered as a rich REPL message (with colors in the web UI) and
  as a JSON object when an environment flag is set.
- The web REPL's error formatter learns the new variant and shows
  a one-line "expected ... got ... at axis ..." summary with the
  offending op name.
- The existing error site in `mlpl-eval::eval_ops` for broadcasting
  failures migrates to the new variant.

### Phase 5: Agent-facing introspection

Wire the label metadata into the existing `:help` surface:

- New `:describe <name>` REPL command: prints the variable's
  labeled shape, dtype, and the first couple of slices. For models
  it prints the `ModelSpec` tree with per-layer labeled shapes on
  the input/output.
- New `:vars` / `:models` / `:fns` REPL commands (scope-local
  versions of APL's `)VARS` and `)FNS`), documented in the web
  help dialog.
- `trace` JSON export includes axis labels so downstream tooling
  can reason about shape flow through a run.

### Phase 6: Docs, tutorial, release

- New tutorial lesson "Named Axes" immediately after "Matrices"
  but before "Linear Algebra", so every subsequent lesson can use
  labels.
- Update the Model DSL lesson and the moons/transformer demos to
  label their inputs and outputs. The Model DSL itself should
  report `[seq, d_model]` inputs and outputs on `apply(mdl, X)`.
- Update `docs/are-we-driven-yet.md`: move Named Axes,
  Shape-Checked Arrays, Structured Errors, and `:describe` from
  CONS to PLAN/HAVE.
- Update `docs/plan.md` to mark Saga 11.5 COMPLETE.
- Bump REPL banners to v0.7.5.
- Rebuild and deploy `pages/`.
- Tag `v0.7.5-named-axes`.

## Planned steps

| # | Slug | Phase | What it delivers |
|---|------|-------|------------------|
| 001 | labeled-shape-type | 1 | `LabeledShape` + `labels()` built-in; no behavior change |
| 002 | label-introduction | 2 | `label(x, [...])`, annotation syntax on assignment |
| 003 | label-reshape-rules | 2 | transpose preserves, reshape clears, `reshape_labeled` added |
| 004 | label-propagation-elementwise | 3 | +, -, *, / propagate and validate labels |
| 005 | label-propagation-matmul-reduce | 3 | matmul, reduce_add, softmax by axis name |
| 006 | shape-mismatch-error | 4 | Structured `ShapeMismatch` error variant + rendering |
| 007 | describe-vars-fns-commands | 5 | `:describe`, `:vars`, `:models`, `:fns` REPL commands |
| 008 | trace-json-labels | 5 | Label metadata in trace JSON export |
| 009 | named-axes-tutorial-lesson | 6 | New lesson + Model DSL lesson refresh |
| 010 | named-axes-release-v075 | 6 | docs, banners, release tag |

Ten steps. Expect some merging once implementation starts --
steps 004 and 005 in particular may collapse if the elementwise
path generalizes cleanly.

## Success criteria

- A labeled array prints its labels alongside its shape in the
  REPL: `X: [seq=6, d_model=4]`.
- A deliberately-wrong matmul (e.g. `matmul(A_labeled[batch, dim],
  B_labeled[time, dim])`) raises a structured `ShapeMismatch`
  error naming the op, the expected contraction axis, and the
  actual labels.
- `:describe mdl` on a stored model prints its layer tree with
  the expected input and output labeled shapes for the first
  layer.
- `:vars` lists every bound array with its labeled shape.
- Every existing demo still runs unchanged: positional arrays
  continue to work without labels.
- `docs/are-we-driven-yet.md` shows at least four rows moving
  from CONS to HAVE.
- All quality gates green; pages deployed; release tagged.

## Risks and open questions

- **Label propagation through broadcasting.** Scalar broadcast is
  easy. Row-vector-against-matrix broadcast is fine if only one
  side is labeled. Two labeled sides with compatible dims but
  different labels is genuinely ambiguous -- should it error, or
  silently prefer the left-hand labels? Default proposal: error,
  with an explicit `relabel(x, [...])` escape hatch.
- **Parser impact of the annotation syntax.** `x : [a, b, c] = ...`
  introduces a colon in assignment that the current parser does
  not expect. Step 002 may require a small parser addition; keep
  it contained so the rest of the grammar is unaffected.
- **`ModelSpec` labeling.** Models are defined before they see any
  input, so their input labels are unknown at construction time.
  Proposal: track labels opportunistically -- the first `apply`
  call pins them, and subsequent `apply` calls with mismatched
  labels raise a structured error. Alternative: require models
  to be constructed with an explicit signature. Decide in step 005.
- **Rank-3+ tensors.** The current op surface is largely rank-2.
  Labels for rank-3+ are trivial metadata-wise, but per-head
  attention (multi-head, currently blocked on slicing) would
  benefit from labeled rank-3 paths. Out of scope for this saga;
  note it as a hand-off to a later attention-slicing saga.
