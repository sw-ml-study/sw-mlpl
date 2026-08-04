# Data-Forge: design for the learning-experience surface

Status: DESIGN FOR REVIEW (saga data-forge, step 1). No code has
shipped; this document proposes the MLPL surface and asks for
sign-off before implementation.

## The thesis

Training-data construction is itself an ML algorithm: generate
candidates, verify them against an oracle, keep the good ones,
order them by difficulty, and record where every example came
from. sw-MLPL should make that whole loop expressible in a few
lines, array-first, with deterministic oracles -- because the
small-model evidence says data quality, validation, and ordering
buy back much of what parameter count gives up.

## Design constraints

1. **Array-first examples.** An example is a row of token ids or
   numbers; a dataset is an `[n, L]` array plus companion arrays
   (`[n]` targets, `[n]` difficulty, `[n]` scores). Text-typed
   examples wait for the strings work; nothing here depends on it.
2. **Evaluators are ordinary code.** An oracle is a `u:` function
   (or a builtin) returning a score; deterministic checks first
   (exact match, graph-path verification, arithmetic identity).
   LLM-judge evaluators arrive with the agent track, not here.
3. **Reuse before invent.** `val_split`, `shift_pairs_*`,
   `gather_rows`, `one_hot`, `scatter`, `sample`, `top_k`, and
   `experiment` tracking already exist and slot straight in.
4. **The selection substrate is the product.** Rejection sampling
   and ranking get reused by MTP proposal analysis, ICRL, and
   distillation later -- so they are built from small orthogonal
   primitives, not a monolithic `rejection_sample()` block.

## Proposed new builtins (nine, all small)

### Ordering + selection (these are APL2 gap G2, pulled forward)

| Builtin | Semantics |
|---|---|
| `grade_up(v)` / `grade_down(v)` | Argsort: the index vector that sorts rank-1 `v` ascending / descending. `gather_rows(X, grade_up(d))` reorders a dataset -- the curriculum primitive. |
| `compress(mask, a[, axis])` | Keep the slices of `a` where rank-1 `mask` is nonzero (APL compress; default axis 0). `compress(gt(scores, 0.8), C)` IS `filter_reward`. |

These two carry most of the substrate: select-best, top-fraction,
class balancing, and staged curricula all compose from
`grade_* + gather_rows + compress + comparison masks`.

### Generation support

| Builtin | Semantics |
|---|---|
| `rand_ints(n, lo, hi, seed)` | `[n]` uniform integers in `[lo, hi)` -- the missing piece for template/grammar/mutation generators (randn exists; uniform ints do not). Deterministic per seed, same bits on every backend. |
| `dedupe_rows(X)` | Unique rows of `[n, L]` (first occurrence kept), returned with the surviving indices so companion arrays follow via `gather_rows`. |

### Minimal knowledge graph (task oracle, not a GNN)

A graph is plain data -- no new value type: entities are integer
ids, and the graph is an `[E, 3]` edge array of
`(src, relation, dst)` rows. Four builtins make it an oracle:

| Builtin | Semantics |
|---|---|
| `kg_neighbors(edges, node[, rel])` | Ids reachable from `node` in one hop (optionally along one relation). |
| `kg_verify(edges, path)` | `1.0` iff every consecutive pair in the rank-1 `path` id sequence is an edge -- the answer checker for multi-hop tasks. |
| `kg_paths(edges, hops, n, seed)` | `[n, hops+1]` valid paths sampled uniformly -- the multi-hop task GENERATOR (question = first id + relation sequence; answer = last id). |
| `kg_split(edges, frac, seed)` | Record `{train, eval}` of edge arrays, split by ENTITY so eval paths visit unseen regions -- the generalization-vs-memorization split. |

## What stays in-language (deliberately)

- **Generators** are `u:` functions built on `rand_ints` +
  `scatter`/`concat`/`mod` (templates, corruptions) or on
  `kg_paths` (graph tasks) or on `sample` (teacher models). No
  generator registry; a generator is just code you can read.
- **Scoring loops**: batch-applying `u:oracle` over rows is a
  `for` loop today. A vectorized `evaluate_rows(:u:f, X, Y)`
  needs first-class user functions (the `:u:` item in the APL2
  higher-order work) and is explicitly deferred to it.
- **Curriculum scheduling** is an idiom, not a builtin:
  `ordered = gather_rows(X, grade_up(difficulty))`, then a stage
  loop trains on `compress(lt(difficulty, stage_max), ordered)`.
- **Provenance** is a Record by convention, stored beside the
  experiment: `prov = {generator: "kg_paths", seed: 7, hops: 3,
  candidates: 10000, accepted: 8123, threshold: 0.9}`. The
  agent-episodes saga later formalizes lineage; data-forge only
  establishes the habit.

## Worked examples (the acceptance demos)

Best-of-N / rejection sampling:

```text
C = u:gen_candidates(10000, 7)          # [n, L] candidate rows
scores = u:score_all(C, Y)              # [n] via the oracle loop
accepted = compress(gt(scores, 0.9), C) # keep verified examples
best = gather_rows(C, take(grade_down(scores), 0, 0))
```

Graph multi-hop curriculum:

```text
g = u:build_toy_graph()                 # [E, 3] edges
split = kg_split(g, 0.8, 7)
tasks3 = kg_paths(split.train, 3, 2000, 11)
ok = u:verify_all(split.train, tasks3)  # kg_verify per row
tasks = compress(ok, tasks3)
idx = grade_up(u:difficulty(tasks))     # e.g. path length / rarity
curriculum = gather_rows(tasks, idx)
```

Both become playground demos; the arithmetic curriculum (digit
tasks from `rand_ints`, verified by evaluating the expression)
is the third.

## Placement

- `grade_up` / `grade_down` / `compress`: `mlpl-runtime-array`
  (they are general array primitives; the APL2-parity docs update
  to "shipped" for these two G2 items).
- `rand_ints` / `dedupe_rows`: `mlpl-runtime-data` beside the
  existing dataset builtins.
- `kg_*`: a new `components/forge/` component,
  `mlpl-forge-kg` crate (its own workspace keeps the disk/test
  blast radius tiny, per the partition policy).

## Open questions for review

1. Naming: `kg_paths` vs `kg_sample_paths`; `dedupe_rows` vs
   `unique_rows`; `compress` (APL heritage) vs `keep`/`filter`
   (plainer English -- but `filter` collides with future HOF
   naming and `compress` matches the documented APL2 gap list).
2. `dedupe_rows` return shape: indices only (compose with
   gather_rows), rows only, or a `{rows, index}` record?
3. Should `kg_verify` be row-batched (`[n, hops+1]` -> `[n]`
   mask) instead of single-path? Batched avoids the scoring-loop
   gap entirely for the graph demos and is barely more code --
   current lean: YES, batched, with rank-1 accepted as `n = 1`.
4. Scope check: is holding generators/curriculum as in-language
   idioms (rather than builtins) acceptable for this saga, per
   constraint 4?
