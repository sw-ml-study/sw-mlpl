# APL2 Feature Staging Plan

A multi-stage plan for adding APL2-inspired array features to MLPL,
prioritized by how much each one illuminates a basic neural-network
forward pass (embed / linear / attention / activation / decode) and by
its value for visualizing the rank, shape, and depth of data.

All feature names are ASCII keyword builtins, consistent with the
project's ASCII-first rule. Unicode glyph aliases (the classic APL
characters) are a later, additive concern -- the same precedent as
`iota` aliasing `range`. This document names glyphs only in prose, for
readers who know the APL heritage.

## Current state (baseline)

MLPL today is a flat, row-major, f64-only dense tensor engine
(`DenseArray` = data vector + shape + optional axis labels), exposed
through Python/ASCII-style keyword builtins. It has a rich ML stack
(autograd, conv/pool/RNN/LSTM/attention, Adam/SGD, LoRA, CPU/MLX/CUDA
backends, PCA/t-SNE/UMAP/MDS) and a 3D "stage" viewer that already maps
array rank to 3D geometry and shows shape/rank/element metadata.

What is missing is essentially the entire APL2 structural layer:
nested arrays (depth greater than 1), the operator algebra (each, rank,
inner/outer product, scan), and the structural verbs (reverse, rotate,
general transpose, take/drop, grade/sort, index-of, membership,
replicate/where). There is also no canonical box-diagram structural
display.

## Priority rationale: which features explain a neural net

A tiny transformer forward pass maps cleanly onto APL2 operations. The
operations that *are* the pipeline are the ones worth demonstrating:

| NN step                        | APL2 operation                          |
|--------------------------------|-----------------------------------------|
| embed (tokens to rows)         | index-of / gather                       |
| linear (X times W)             | inner product (matmul is add-dot-mul)   |
| attention scores               | outer product, then inner product       |
| activation / softmax-per-row   | each, and the rank operator             |
| decode / sample                | grade / sort, top-k                     |
| head split, windowing          | dyadic transpose, take/drop, reverse    |
| see it all                     | disp box display, depth                 |

Highest leverage, in order: disp/depth, then each/cells (rank
operator), then inner/outer product, then structural transforms, then
index-of/grade. Every one of these is flat-array-only (no data-model
change) except true depth, which is deliberately last.

## Shared plumbing (every stage)

- Register the builtin in `components/types/crates/mlpl-eval-core/src/inspect_groups.rs` (BUILTIN_GROUPS).
- Dispatch in `components/eval/crates/mlpl-eval/src/eval_fncalls.rs`. Operators follow the `eval_reduce.rs` template (they take a BuiltinRef `:op`).
- Kernels live in `components/array-element` and `components/array-compose`.
- Demo `.mlpl` file under `demos/`, registered in `components/web-demos/crates/mlpl-web-demos/demos.toml`.
- Tests under `components/eval/crates/mlpl-eval/tests/` plus kernel-local unit tests.
- Each stage is several small commits (one verb/operator plus its test
  and demo per commit) to fit the sw-checklist ratchet and TDD gate.
  The three mapping operators (each, cells, scan) share one
  "apply a BuiltinRef over cells" helper -- define once, invoke many.

Narrative arc across the stages: see shapes, apply per-cell, combine,
reshape, encode/decode, nest.

## Stage 1 -- See the shapes (structural introspection)

The enabler reused by every later demo, and the direct answer to
"visualize rank/shape/depth."

- Features: `depth(x)`; `disp(x)` text box diagram (shape on the frame
  edges, rank as nested frames, axis labels shown); a new
  `svg(_, "structure")` diagram type; shape-edge labels plus a depth
  badge on the 3D stage (`stage3d.js` shapeMesh,
  `components/web-viz3d/.../events.rs`).
- Demo: `structure_zoo.mlpl` -- scalar, vector, matrix, 3-tensor, each
  passed through `disp`; then `disp` on a real weight matrix and an
  activation.
- Tests: golden text of `disp` per rank; `depth` returns 0 for a
  scalar and 1 for any flat array; SVG kind and size snapshot.
- NN story: connective tissue -- watch every intermediate shape.

## Stage 2 -- Apply per cell (mapping operators)

- Features: `each(:f, x)` (map a function over cells);
  `cells(:f, k, x)` (the rank operator: apply f to each rank-k
  subarray); general `scan(:op, x[, axis])` (generalizes the existing
  `cumprod`). One shared apply-over-cells helper backs all three.
- Demo: `activation_zoo.mlpl` -- `each(:relu, X)` and
  `each(:sigmoid, X)` as before/after heatmaps; `cells(:softmax, 1,
  logits)` = "softmax every row"; `scan(:add, x)` = cumulative mask.
- Tests: `each` equals the elementwise builtin; `cells` with k=1 on a
  matrix equals a row-wise apply; `scan` equals the prefix reduce;
  rank and shape preserved.
- NN story: activations and softmax-per-row/head. The rank operator is
  the star: apply softmax to every attention row.

## Stage 3 -- Combine two arrays (product operators)

- Features: `outer(:f, a, b)` (outer product); `inner(:f, :g, a, b)`
  (generalized inner product).
- Demo: `products.mlpl` -- `outer(:mul, iota(n), iota(n))` = a
  multiplication-table heatmap, then `outer(:mul, q, k)` toward
  attention scores; assert `inner(:add, :mul, A, B) == matmul(A, B)`
  ("matmul is just an inner product").
- Tests: `outer` result shape equals shape(a) concatenated with
  shape(b); `inner(:add, :mul)` matches `matmul` on random matrices;
  `outer(:mul)` matches the broadcast product.
- NN story: the linear layer and attention scores demystified.

## Stage 4 -- Reshape the data flow (structural transforms)

- Features: general dyadic `transpose(x, perm)` (today it only reverses
  all axes); `reverse` and `rotate`; `ravel` / flatten; dyadic `take`
  and `drop` with overtake fill.
- Demo: `reshape_flow.mlpl` -- split heads via `transpose(x, [1,0,2])`
  plus `reshape`, animated on the 3D stage as a re-tiling; `take` /
  `drop` to crop a batch; `reverse` a sequence.
- Tests: `transpose(perm)` matches a known permutation; `ravel` then
  `reshape` round-trips; `take` / `drop` boundary and overtake cases.
- NN story: multi-head head-splitting and windowing -- the plumbing
  between layers made visible.

## Stage 5 -- Encode and decode (selection, ranking, lookup)

The capstone stage: it composes the earlier pieces into the full
encode/decode pipeline.

- Features: `index_of(x, y)`; `member(x, y)`; `grade_up` and
  `grade_down` plus `sort`; `where` / `replicate`.
- Demo: `encode_decode.mlpl` -- encode: text to `index_of` into the
  vocab to token ids (visualized as a gather); decode: logits to
  `grade_down` to top-k to detokenize. Plus a `narrated_transformer.mlpl`
  capstone: a full tiny forward pass narrated entirely in APL2 ops
  (disp, each/cells, inner/outer, index-of, grade), every intermediate
  visualized.
- Tests: `index_of` matches a linear scan; `grade_down` matches
  argsort; `sort` is stable; `member` yields the correct mask; the
  capstone has a golden run.
- NN story: the tokenizer/embedding front-end and the sampling
  back-end -- encode and decode, end to end.

## Stage 6 -- Depth (nested arrays)

The only invasive, data-model-changing stage, so it is isolated and
last. This is the stage that makes APL2 depth real and visualizable.

- Features: a boxed cell type (a Value::Boxed variant, or a boxed
  DenseArray element -- `components/eval-types/.../value.rs`,
  `components/array/.../dense.rs`); `enclose` / `disclose` / `pick` /
  `first`; `depth` becomes variable; `each` now maps over boxes; `disp`
  renders nested frames.
- Demo: `ragged_batch.mlpl` -- variable-length token sequences as a
  nested array, passed through `disp` as nested boxes;
  `each(:embed, batch)` over the ragged rows.
- Tests: `depth` increments on `enclose`; `disclose` inverts it; `pick`
  indexes; ragged shape reporting; nested-box `disp` golden output.
- NN story: variable-length sequences and lists-of-tensors -- depth is
  finally real and visualizable.

## Ordering summary

1. Stage 1 is the tool every other demo leans on and directly answers
   "visualize rank/shape."
2. Stages 2 and 3 are the highest-illumination features for a forward
   pass.
3. Stage 4 is plumbing that Stages 2 and 3 make meaningful.
4. Stage 5 composes everything into the explicit encode/decode capstone.
5. Stage 6 is the only data-model change, so it is isolated and last.

## Motivating capstone: Conway's Game of Life

`docs/future-saga-game-of-life.md` (2026-07-26) audits the classic
one-liner against these stages: Life runs TODAY on the flat subset
(permutation-matmul shifts, verified live), Stage 3 + 4 make it read
like the APL, and the Stage 6 nested-array version (enclosed board,
outer-rotate over the box, strand inner product) is the natural
acceptance demo for Stage 6 itself. The audit also surfaced two
stage-adjacent gaps: operators should accept `u:` function operands
(not just builtins), and there is no indexed/selective assignment
(`put`/scatter) -- both worth folding into Stage 2/5 scoping.
