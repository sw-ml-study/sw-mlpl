# Correctness and Performance: MLPL language gaps

Status: OPEN. Captured 2026-06-02 while implementing a tic-tac-toe
minimax engine *in MLPL* (the right way to showcase the array language,
replacing the domain-specific `ttt_boards` / `ttt_moves` /
`play_vs_random` Rust builtins). That work surfaced language bugs and a
performance wall that block writing real recursive / array-heavy
algorithms in MLPL. This doc records them and the sagas that fix them.

Tracking issues:

- Correctness: <https://github.com/sw-ml-study/sw-mlpl/issues/6>
- Performance: <https://github.com/sw-ml-study/sw-mlpl/issues/7>

## Why this matters

sw-mlpl is an APL/J/BQN-lineage array language. A demo that hides the
game logic (board, winner, minimax, encoding) inside Rust builtins fails
to demonstrate the language's value proposition. The proper demo
*expresses* the engine in MLPL:

- winner detection as a `matmul` with a constant `[8,9]` line-incidence
  matrix (`M @ board` = the eight line-sums) -- this works today;
- legal moves as a boolean mask;
- the optimal policy as a recursive `minimax` (`FnDef` + `If` + `For`);
- the board encoding as elementwise array ops;
- self-play as a `While` loop.

The AST already has `If` / `While` / `For` / `Repeat` / `FnDef` /
`Return` / records, and `FnDef` registers before its body runs (so
recursion resolves). So the engine *is* expressible -- except for the
two correctness bugs and the performance wall below.

## Correctness issues (issue #6)

### C1 (HIGH): user-function calls share caller locals (no per-call scope) -- FIXED (c40f9a68)

Fixed: `call_user_fn` now snapshots the variable namespaces and restores
them on return (per-call scope); `env_scope.rs`. Repro now returns
`3 5 1`; regression tests in `mlpl-eval/tests/fn_scope.rs`.

Original report:

A recursive call overwrites the caller's locals, so a function that
binds a local, recurses, then reads that local gets the deepest frame's
value.

Repro:

```
def u:f(n) { if gt(n, 0) { keep = n; tmp = u:f(n - 1); keep } else { 0 } }
[u:f(3), u:f(5), u:f(1)]
```

Expected `3 5 1`; actual `1 1 1`.

Impact: any recursion that accumulates / reads a local across a
recursive call is silently wrong. The tic-tac-toe minimax `score(board,
player)` (loops over legal moves accumulating `best` while recursing)
returns garbage -- empty-board minimax returns `1` instead of the
correct draw value `0`.

Likely fix: `call_user_fn`
(`components/eval/crates/mlpl-eval/src/eval_user_fn.rs`) must evaluate
the body in a fresh child variable frame and not leak local assignments
into the caller / sibling frames. Reads of outer variables may remain
(documented "access to outer variables"), but writes must be local.

### C2 (limitation): newlines forbidden inside array literals -- FIXED (aeb480e2)

Fixed: `parse_array_lit` now skips newline tokens (`skip_newlines`)
around elements/commas, so a matrix can span lines. Tests in
`mlpl-parser/tests/multiline_array_tests.rs`.

Original report:

Repro:

```
x = [1, 2,
3, 4]
x
```

Expected the 4-vector `1 2 3 4`; actual `error: unexpected token
'newline'`. Cannot format a matrix across lines (e.g. an 8x9 incidence
matrix). Newlines inside `[ ]` (ideally `( )` too) should be
insignificant.

### C3 (minor): bare `if` without `else` as a statement

`if gt(x, 0) { x }` on its own line errors at the trailing newline (`if`
parses as an expression requiring `else`). Either support
statement-position `if` without `else`, or emit a clearer diagnostic.

## Performance issue: interpreted minimax is ~10.8s (empty board)

A correct MLPL minimax from the empty board takes ~10.8s; Rust is
instant. The full labeled policy dataset (~4520 positions) is therefore
impractical to generate at runtime in the interpreter.

Causes:

1. **Tree-walking interpreter vs native.** Minimax from empty visits
   ~549k game-tree nodes; each node runs several ops (the `winner`
   matmul + reshape x2 + eq x2 + reduce_add x2). Millions of interpreted
   dispatches; Rust compiles to machine code.
2. **Heap allocation per op.** Every op returns a fresh heap
   `DenseArray`. One `winner` ~= 7 allocations; `board + player *
   eq(iota(9), c)` allocates per legal move per node. Rust's `Board` is
   a stack `[i8; 9]` -- zero allocation.
3. **Recomputed constants.** `iota(9)` / `eq(iota(9), c)` rebuild arrays
   every node/iteration.
4. **No pruning or memoization.** Full tree; shared subtrees re-derived
   ~549k times.

Fixes, highest leverage first:

- **P-A. Memoize (transposition table).** Tic-tac-toe has only ~5,478
  distinct states; evaluate each once (~100x: 549k -> 5.5k). Needs a
  dict / hashmap primitive keyed by a board hash (real language gap --
  records are fixed-field).
- **P-B. Alpha-beta pruning.** 10-100x, no new primitive; free once C1
  is fixed.
- **P-C. Vectorized DP over all states.** Enumerate states as `[N,9]`,
  batch winners via one `[8,9] @ [9,N]` matmul, propagate minimax values
  ply-by-ply with array reduces. Replaces 549k recursive calls with a
  handful of big ops -- fast AND the idiomatic array-language showcase.
- **P-D. Interpreter allocation reduction.** Hoist constants out of hot
  paths; small-array reuse / arena; scalar fast-path. ~2-5x constant
  factor.
- **P-E. Cache static labels.** Minimax output never changes; compute
  once, reuse. Trivial, but does not exercise the search.

Best combination: C1 fix + P-B (alpha-beta) + P-A (memo dict) drops
minimax to ~5.5k evaluations, fast even interpreted. P-C is the most
elegant showcase if leaning into the array-language angle.

## Sagas

### Saga: language-correctness (prerequisite)

Fix the correctness bugs that make recursion / array literals usable.

1. **fn-scope** -- DONE (c40f9a68). `call_user_fn` snapshots/restores
   the variable namespaces per call; local writes do not leak to caller
   / siblings; outer reads still work. Regression tests in
   `fn_scope.rs` (recursion-with-locals, factorial, no-leak). (Fixed C1.)
2. **multiline-array** -- DONE (aeb480e2). `parse_array_lit` skips
   newlines inside `[ ]`; a multi-line matrix parses. Tests in
   `multiline_array_tests.rs`. (Fixed C2. Call-arg `( )` newlines remain
   a future nicety, not needed for matrix literals.)
3. **bare-if-diagnostic** -- either accept statement-position `if`
   without `else`, or produce a precise diagnostic pointing at the
   missing `else`. (Fixes C3.)

### Saga: interpreter-and-search-performance

Make recursive / array-heavy MLPL fast enough for real algorithms.

1. **alpha-beta** -- once C1 is fixed, add alpha-beta to the MLPL
   minimax pattern (demo-side) and document the idiom. (P-B.)
2. **memo-dict-primitive** -- add a dict / hashmap value + builtins
   (`dict`, `get`, `set`, `has`) keyed by an array hash, enabling
   transposition tables and dedup (also unblocks `unique`). (P-A.)
3. **interp-alloc** -- hoist loop-invariant constants, reuse small-array
   buffers / arena intermediates, add a scalar fast-path in the op
   dispatch. Bench minimax before/after. (P-D.)
4. **vectorized-dp** (stretch) -- a batched all-states minimax DP as the
   array-idiomatic reference, plus a `where` / `nonzero` primitive
   (boolean mask -> indices) used by `legal`. (P-C, plus the legal-moves
   gap.)

### Saga: tictactoe-mlpl-engine (depends on both above)

Replace the `ttt_boards` / `ttt_moves` / `play_vs_random` Rust builtins
with MLPL definitions (`winner` via the incidence matmul, `legal`,
`encode`, `minimax`, dataset enumeration, self-play loop), update the
literate page + web demo to define and use them, and remove the builtins
from `mlpl-eval`. Blocked on language-correctness (recursion) and
interpreter-and-search-performance (so the dataset generates in seconds).

## Interim state

Until the above land, the tic-tac-toe demo keeps the Rust builtins
(`ttt_boards`, `ttt_moves`, `play_vs_random`) as a working fast path;
the literate page shows the engine's Rust source as explained listings.
This is a documented stopgap, not the end state -- the end state is the
MLPL engine above.
