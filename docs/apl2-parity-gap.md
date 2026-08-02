# APL2 feature parity: gap inventory and priorities

Recorded 2026-08-01 (user direction). The dual motivation, stated
once so every future saga can cite it:

> sw-MLPL is a MODERN APL2 ADAPTED TO MODERN ML. ML use cases are
> the first priority. General-purpose programming -- the classic
> APL2 application space -- is a secondary goal, but it GATES what
> "complete" means for the language, and demonstrating some
> backward-compatible general programming is part of the project's
> historical thesis.

Naming ground rule (landed with the iota deprecation): meaningful
ASCII names are canonical and arity-locked one-to-one; APL glyph
names are heritage aliases at most. No math symbols or Greek
letters in the primary surface.

## What already holds up against APL2

Array core (rank-N f64 arrays, axis labels, broadcasting,
elementwise ops, matmul/dot, reshape/transpose/rotate/take/
flatten, reductions incl. the higher-order `reduce(:op, a[, axis])`),
the quoted-builtin reference `:name` (APL's quoted function), the
`u:` user-function namespace with recursion and control flow
(if/while/for/repeat), boxed display (`disp`), functional lenses in
place of selective assignment (docs/functional-lenses.md), the
Result/attempt error-as-data design (docs/error-handling.md), and
the deliberate OMISSION of execute (no eval-string -- the
compile-to-Rust enabler, docs/milestone-compile-to-rust.md).
Game of Life and tic-tac-toe minimax -- classic APL showpieces --
already run.

## Gap inventory

### G1. Function composition + higher-order functions (USER PRIORITY)

The seeds exist (`:name` values, `reduce(:op, ...)`, variables
holding BuiltinRefs) but stop short of APL2's operator algebra.
Wanted, in compact ASCII:

- `each(:f, a)` -- APL's `f each` / map over major cells (and over
  elements for rank-0 f). The single most-used APL2 operator.
- Composition VALUES: `g_of_f = :f >> :g` (apply f, then g) --
  an ASCII operator, no jot/compose glyph. Alternative names
  considered: `compose(:g, :f)` fn form as day one, `>>` infix as
  the compact sugar once `Value::Function` lands.
- Application pipe for readability at call sites:
  `x |> u:clean |> u:tokenize |> shape` (left-to-right data flow;
  ASCII, no trains).
- Partial application / binding: `pow2 = bind(:pow, _, 2)` with
  `_` as the hole marker.
- `outer(:op, a, b)` -- APL2 outer product (jot-dot); `inner(:f,
  :g, a, b)` -- generalized inner product (f.g). `matmul` is the
  `inner(:add, :mul, ...)` special case.
- `scan(:op, a[, axis])` -- APL2 scan (running reduce; cumprod is
  the one special case shipped). The scan saga should also carry
  the pedagogy: a scan is the ASSOCIATIVE-recurrence borderline
  between data loops (which array ops absorb) and time loops
  (which survive as `repeat`/`train`) -- the "Thinking in Arrays"
  playground demo sets this up with cumprod and points here.
- Prerequisite: user functions as first-class values (`:u:name`,
  already planned in the lang-reference "symmetry to come" note)
  so every HOF accepts builtins AND user functions uniformly.

ML payoff: each/outer/scan are batch-shaped ML operations;
composition values are model/data pipelines; `bind` is config
currying. This gap serves BOTH priorities at once.

### G2. Ordering and selection

- `grade_up(a)` / `grade_down(a)` -- argsort indices; `sort(a)` as
  the convenience. ML needs this constantly (top-k beyond argtop_k,
  beam search, sampling without replacement, calibration curves).
- `compress(mask, a, axis)` / boolean selection, and `expand`.
- APL-style `take`/`drop` of leading cells (head/tail windows) --
  MLPL's `take(x, axis, idx)` is single-index extraction, a
  different op.
- General indexing by index ARRAYS (gather_rows exists for rank-2
  rows; the general form does not).

### G3. Strings as data (the biggest general-purpose blocker)

MLPL strings are opaque atoms: no split/join/substring/find/
replace/case, no char-code conversion, no string comparison
beyond equality, no formatted output (APL2 format / picture
format). Classic APL2 report programs are IMPOSSIBLE today, and
ML data prep (corpus cleaning before tokenize_bytes/BPE) suffers
the same gap.

### G4. Nested / general arrays

APL2's defining feature: arrays whose items are arrays (enclose/
disclose, depth > 1, each distributing over items). MLPL's `depth`
builtin already anticipates this ("higher once nested arrays
land"). Blocks: ragged batches (ML), record-like data, most
classic APL2 idioms. This is the largest single parity item and
deserves its own design saga (interaction with labels, the tape,
and serialization all nontrivial).

### G5. General-purpose runtime surface

- File I/O from the language (read/write text + arrays; datasets
  currently arrive via dedicated builtins like load_preloaded /
  fetch_dataset only).
- Interactive input (APL's quad/quote-quad prompt loop) -- needed
  for classic menu-driven programs; web REPL equivalent = a
  prompt() that suspends.
- Date/time functions.
- Formatted numeric output (column alignment, precision control)
  -- pairs with G3.
- Complex numbers (APL2 has them; ML need is low -- FFT-adjacent
  work only; keep LAST).

### G6. Already-planned adjacent items

First-class user functions + `attempt(u:f, u:handler)` (staged),
selective assignment as lens sugar (functional-lenses.md), `:u:`
quoting symmetry.

## Litmus tests: classic APL2 apps sw-MLPL cannot host yet

The question "what traditional APL2 program can't be written?"
has a crisp answer -- anything string-heavy, record-keeping, or
interactive:

1. A formatted sales-ledger / report generator (nested string
   arrays, grade_up on keys, picture-format columns) -- blocked by
   G2 + G3 + G4 + formatted output.
2. An inventory or address-book application (records as nested
   arrays, key lookup, file persistence between sessions) --
   blocked by G3 + G4 + G5 file I/O.
3. A menu-driven interactive utility (quad-input loops) -- blocked
   by G5 input.
4. A text-adventure or parser-driven game -- blocked by G3.
5. A date-arithmetic utility (workdays between dates) -- blocked
   by G5 dates.

By contrast, numeric/matrix classics (Life, minimax games, stats
tables, simulations) already work -- the array core is not the
gap; the DATA DIVERSITY (text, nesting, records) and the OPERATOR
ALGEBRA are.

## Historical note: APL and early ML

Part of the project's thesis. APL sits in the prehistory of ML
computing: IBM's APL\360 era made interactive matrix computation
available decades before Python; perceptron-era and
statistics-heavy work (regression, factor analysis, clustering)
was routinely done in APL in the 1970s-80s because inner/outer
products, reductions, and grade were one-liners; IBM research
groups prototyped neural-net experiments in APL2 in the late
1980s when backprop revived, precisely because `inner(:add,:mul)`
IS the forward pass. The lineage runs APL -> S/MATLAB -> NumPy:
today's `ndarray` broadcasting semantics are APL's, renamed. The
demo-worthy story: an MLPL notebook that first runs a classic
1970s-style APL2 program (ledger or Life) and then trains a
transformer with the SAME array vocabulary.

## Priority mapping (ML-first ordering of the gaps)

1. G1 composition/HOFs + G2 grade/sort -- serve ML immediately
   (pipelines, top-k/beam, batch ops) AND unlock the operator
   algebra. First parity saga.
2. G3 strings -- unlocks data prep for ML and the report-class
   apps. Second saga.
3. G4 nested arrays -- ragged batches for ML; the big design
   saga; schedule after the current E4/E5 + speed-track work has
   an owner.
4. G5 general-purpose runtime -- completeness gate items; batch
   as a "classic apps" saga whose acceptance test IS litmus apps
   1-5 running.
5. Complex numbers -- last.

Queue placement: see docs/future-sagas-queue.md (these slot into
the maintenance/track structure without displacing the E4/E5 ->
generation-speed -> agent-quality spine).
