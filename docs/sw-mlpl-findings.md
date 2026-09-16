# sw-mlpl dogfooding findings (from ../moe-microscope)

Status: RESOLVED (audit trail). Findings surfaced by the moe-microscope
repo agent probing mlpl-repl 0.22.0 while building the MoE educational
visualization demo (dogfooding sw-mlpl + demo-extensions, not just proving a
model trains). All four were addressed in the `moe-microscope-findings` saga;
each finding below keeps its original reproducer and carries a RESOLVED line
with the fixing commit. Reproduced against 0.22.0; fixes verified in the
rebuilt release binary.

Resolution summary:

| Finding | Severity | Resolution | Commit |
|---------|----------|------------|--------|
| F1 softmax arity | HIGH | `softmax(x)` defaults to the last axis eagerly, matching the tape | `ae7ec747` |
| F2 user fns in grad | HIGH | `u:` calls are inlined onto the grad tape | `41faf626` |
| F3 chain sharing | MEDIUM | documented: chain copies blocks; reuse one block to share | `b28fdf45` |
| F4 gather_rows / KL | LOW | `gather_rows` differentiable (scatter-add); KL documented as a composition | `e1d693af` |

## F1 -- softmax arity differs between eager and tape (HIGH)

Eager evaluation requires an axis; inside `grad`/`adam` (tape) it takes one
argument. The same loss expression cannot be written once and reused for both
evaluation and training.

```
softmax([1.0, 2.0, 3.0])       # error: softmax: expected 2 args, got 1
softmax([1.0, 2.0, 3.0], 0)    # ok -> 0.090..., 0.244..., 0.665...
```
(The moe agent reports the tape path requires the 1-arg form.)

Proposed fix: accept an OPTIONAL axis in both eager and tape (default the last
axis), so `softmax(x)` and `softmax(x, axis)` mean the same in both contexts.
Real bug or at least a doc gap; pick one arity and honor it everywhere.

RESOLVED (`ae7ec747`): eager `softmax(x)` now defaults to the last axis, so a
one-argument softmax means the same thing in evaluation and inside
`grad`/`adam`. `softmax(x, axis)` is unchanged.

## F2 -- user-defined functions rejected inside grad/adam (HIGH; most important)

A `def u:...` cannot be called inside `grad`/`adam`; the loss must be written
inline. This fights the whole point of a language that expresses models as
auditable, reusable source.

```
def u:moe(x) { ... }
grad(u:moe(w), w)   # error: grad: 'u:moe' not supported inside grad()
```
(A minimal `grad(u:sq(w), w)` probe here surfaced the adjacent
"'w' is not a tracked parameter" wall; the moe agent's scratchpad has the
`u:moe` reproducer.)

Proposed fix: trace through user-function calls inside `grad` (inline/expand
the body onto the tape), so a loss defined as `def u:loss(...)` trains.

RESOLVED (`41faf626`): a `u:` call reached inside `grad`/`adam`/`train` is
inlined onto the same reverse-mode tape -- its parameters bind to the traced
arguments and its body is walked statement-by-statement, so it differentiates
exactly as the inline expression. A global param the body references stays
differentiable; a depth guard turns runaway recursion into an error.

## F3 -- chain(blk, blk, blk) does not share weights (MEDIUM; workaround exists)

`chain` triple-counts `param_count` and `adam` fails with "not a tracked
parameter". Nested `apply(blk, apply(blk, apply(blk, x)))` IS the working
spelling for weight-shared recurrence.

Proposed fix: either make `chain` share the block's weights (single tracked
parameter set) or document clearly that `chain` does NOT share and nested
`apply` is the recurrence/weight-sharing spelling.

RESOLVED (`b28fdf45`): documented (by design, chain composes independent
blocks). lang-reference + glossary now state each chain argument is an
independent block whose params are summed; weight sharing / recurrence is
spelled by reusing one block via nested `apply` (or, per F2, a user function),
where the tape accumulates gradient across every use. Pinned by two tests in
`model_dsl_tests` (3x param count; gradient accumulation on reuse).

## F4 -- gather_rows not differentiable; no kl_divergence (LOW; convenience)

`gather_rows` is not on the autograd tape (so a from-scratch Engram addressing
lesson is forward-only). `kl_divergence` is absent:

```
kl_divergence([0.5, 0.5], [0.5, 0.5])   # error: unknown function: kl_divergence
```
KL is a one-line composition of `softmax` and `log`, so this is a convenience
gap, not a blocker.

Proposed fix: add a `gather_rows` backward (scatter-add, like `windows`); add a
`kl_divergence` builtin OR document the composition.

RESOLVED (`e1d693af`): `gather_rows` is now differentiable wrt the table inside
`grad` -- the addressed rows are gathered by a one-hot selection matmul whose
backward is an exact scatter-ADD into the addressed rows (duplicates
accumulate, unaddressed rows stay zero), so a from-scratch addressing /
embedding table trains. `kl_divergence` needs no builtin: it is a direct
composition of shipped builtins that works both eagerly and through `grad` --
`reduce_add(P * (log(P) - log(Q)))`, gradient `-P / Q`. Documented in
lang-reference (Autograd) and the glossary KL entry.

## Triage (historical; all resolved above)

- F1, F2 were the strong candidates for a dedicated saga -- they blocked
  writing a model as reusable source and reusing one loss expression. F2 was
  the most important for the "models as auditable source" goal.
- F3 was documentation (nested `apply` is the intended spelling).
- F4 was a small additive pair (a scatter-add gather backward + a documented
  KL composition; no new builtin needed).

None blocked the moe-microscope Saga 1 or 2; each was a ledger entry with a
reproducer, now closed.

## Follow-up batch 1 (2026-09-12 rerun by ../moe-microscope) -- RESOLVED

The downstream agent re-verified F1-F4 against the rebuilt release binary
(all confirmed) and surfaced two new edges plus one UX note. All three were
addressed in the `moe-microscope-followups` saga and re-verified downstream.

| Finding | Resolution | Commit |
|---------|------------|--------|
| F5 index/mask builtins in grad | stop-gradient constants on the tape | `54ad5849` |
| F6 `repeat` in a traced function | unrolled onto the tape | `a9ae2a79` |
| D1 silent-zero grad | loud error when the loss does not depend on `wrt` | `e505784c` |

### F5 -- index/mask builtins rejected inside a traced function (LOW; workaround)

`one_hot` (and `argmax`) cannot be used inside `grad`, so a top-1 router mask
cannot be computed inside the loss.

```
grad(sum(one_hot(argmax(W), 3) * W), W)
# error: grad: function 'one_hot' not supported inside grad()
```

Workaround: compute the mask eagerly each step and pass it in as a constant
argument -- which is what a Switch-style top-1 gate wants anyway. Upstream fix
would be a stop-gradient treatment of the index/mask builtins on the tape
(they are non-differentiable by nature; the gradient should flow through the
selected values, not the indices).

RESOLVED (`54ad5849`): `argmax`, `one_hot`, `eq`, `gt`, `lt`, `argtop_k` are
now stop-gradient constants inside `grad` -- computed from the current forward
values and inserted as non-tracked leaves. A top-1 router mask
`sum(one_hot(argmax(R, 1), E) * R)` trains, gradient flowing to the selected
logits, never the mask.

### F6 -- `repeat` not usable inside a traced function (LOW; workaround)

Recurrence depth cannot be spelled with `repeat` inside `grad`; `repeat` is a
statement form, not a tape-expressible expression.

```
grad(sum(repeat 3 { W }), W)   # parse / grad rejection
```

Workaround: spell recurrence as nested `apply` (or, per F2, a user function
that applies a block repeatedly) -- so depth `R` is a source-level choice.
One small user function per depth covers a lesson's `R` range.

RESOLVED (`a9ae2a79`): a `repeat N { ... }` inside a traced user function is
unrolled onto the tape -- the body's assignments thread through the local
scope across iterations -- so bounded recurrence trains. (A count bound to a
FUNCTION PARAMETER rather than a global is still unresolved; see F15.)

### D1 -- grad wrt a tape constant returns zeros silently (UX note)

When the `wrt` operand is a constant on the tape (e.g. the loss or the
operand was built from eagerly-assigned values rather than a `param`/`tensor`
leaf), `grad` can return an all-zero gradient with no error, which reads as a
training bug. In the adjacent probes the common shapes error loudly
("'T' is not a tracked parameter"), but a silent-zeros path exists. A loud
error (or a prominent reference note) when the `wrt` leaf is untracked would
save a learner the debugging hour. Aligns with the "capability tests pin the
requirement" and loud-failure conventions.

RESOLVED (`e505784c`): the user-facing `grad(expr, wrt)` now errors when no
gradient reaches `wrt` (the loss does not depend on it) instead of returning
silent zeros, with a message naming the likely causes. The optimizer path
keeps returning zeros for untouched params (correct batched semantics).

## Follow-up batch 2 (2026-09-12 second rerun by ../moe-microscope) -- RESOLVED

A deeper dogfooding pass (the from-scratch TinyMoE "this domain" lesson: a
3,812-param block over a 120-example fixture) surfaced seven more findings,
each with a reproducer in the downstream `probes/`. All were addressed in the
`moe-microscope-followups-2` saga (F18, a second panic reported mid-saga, was
folded in). Both panics (F10, F18) now produce clean MLPL errors or train.

| Finding | Severity | Resolution | Commit |
|---------|----------|------------|--------|
| F9 batched `[B,T]` embed | med | eager + tape flatten-lookup-reshape | `042d2d79` |
| F10 sinusoidal tape panic | BUG | per-axis label unification (no panic; trains) | `9cd62367` |
| F11 nested-fn index arith | low | gather index resolves in the traced scope | `813526d6` |
| F12 shape-derived size | low | constant-fold value-independent subexprs | `82e55a6a` |
| F13 attention_weights in residual | low | recurse into wrapped Attention layers | `8282cd6a` |
| F14 constant constructors in grad | low | `fill`/`zeros`/`ones` as constant leaves | `61da67ae` |
| F15 repeat param-bound count | low | count resolves in the traced scope | `369cfeb1` |
| F18 shape mismatch panic | BUG | pre-validate broadcast/label -> clean error | `9e699317` |

- **F9** -- `embed` rejects the batched `[B, T]` token input the reference
  documents (only `[T]` is accepted). Blocks batched embedding lookups.
- **F10** (BUG, highest priority) -- a labeled `sinusoidal_encoding` table
  PANICS the autograd tape inside a residual block (a Rust panic, not an MLPL
  error). A panic is never acceptable; must become a clean error at minimum,
  ideally supported.
- **F11** -- a nested user-function call inside `grad` loses a parameter used
  in index arithmetic (an F2 inliner edge: a param referenced only through an
  index computation is dropped).
- **F12** -- shape-derived size arithmetic is rejected inside `grad` (deriving
  a size from a tensor's shape to feed a reshape/constructor in the loss).
- **F13** -- `attention_weights` cannot see inside `residual(chain(...))`, so
  the lesson writes the residual by hand (`h1 = h0 + att(h0)`). (The hand form
  is clearer for the microscope and will stay even after a fix.)
- **F14** -- `fill` and similar constant constructors are rejected inside
  `grad` (`grad(sum(W * fill([3], 2.0)), W)` errors). They should be constant
  leaves on the tape (same treatment as literals). Verified against the
  release binary.
- **F15** -- `repeat` with a count bound to a FUNCTION PARAMETER fails inside a
  traced function (`repeat r { ... }` where `r` is an arg -> "undefined
  variable: r"). The F6 unroll resolves the count via the eager env, not the
  traced local scope. Verified against the release binary.

## Follow-up batch 3 (2026-09-12 host-handoff step by ../moe-microscope) -- RESOLVED

Recording the DN01 baseline over a live `mlpl-serve` SSE session surfaced a
server-surface gap and a build-hygiene gap. Addressed in the
`moe-microscope-followups-3` saga.

| Finding | Resolution | Commit |
|---------|------------|--------|
| F16a include over the wire | eval request `includes` map -> MemoryProvider + expand | `902f14e2` |
| F16b filesystem sandbox | `--fs-root` sets each session's `env.fs_root` | `855fb551` |
| F16c args | eval request `args` -> `args()` builtin | `f85cc657` |
| S1 stale serve / rebuild hygiene | serve rebuilt + rebuild-on-evaluator-change discipline | (process) |

- **F16** -- the `eval_stream` server surface has no `include`, no filesystem
  sandbox, and no `args`, so a lesson split into library modules cannot be
  submitted as written; the downstream bundles the include tree inline as a
  workaround. This is sw-mlpl's server surface, so it belongs here -- a real
  feature (include resolution over the wire + a sandbox policy + args passing),
  queued as `moe-microscope-followups-3`.
- **S1** -- the adjacent `mlpl-serve` release binary was stale (it predated the
  user-function-loss support) because only `mlpl-repl`/`mlpl-build` were being
  rebuilt on evaluator changes. PARTLY ADDRESSED: `target/release/mlpl-serve`
  has been rebuilt from current source, and "rebuild `mlpl-serve` whenever the
  evaluator changes" is now part of the checkpoint discipline (it embeds
  `mlpl-eval`). The remaining ask -- a documented/scripted current-server build
  so a fresh clone has a correct server -- rides with F16's saga.

RESOLVED (`902f14e2`, `855fb551`, `f85cc657`): F16 shipped in three parts --
the eval request accepts an `includes` map (resolved via the in-memory source
provider under the same relative-only/no-escape sandbox), a `--fs-root` flag
gives server-run programs a filesystem sandbox root for the fs builtins, and an
`args` field feeds the `args()` builtin. S1's serve rebuild is done and the
rebuild-on-evaluator-change discipline is adopted.

## Follow-up batch 4 (2026-09-13 audit + RM prep by ../moe-microscope) -- RESOLVED

An optimizer-state audit (one training run per process is now the downstream
rule) plus more grad-surface probing surfaced four findings, each with a
reproducer in the downstream `probes/`. F19 and F20 were process panics; F21 and
F22 silent-wrong-training footguns. All shipped in `moe-microscope-followups-4`.

| Finding | Severity | Resolution | Commit |
|---------|----------|------------|--------|
| F19 matmul shape panic in grad | BUG | pre-validate matmul shapes -> clean error (matmul analogue of F18) | `bbcf59dc` |
| F20 take index in grad | BUG | index resolves in the traced scope; out-of-range is a clean error | `e6070964` |
| F22 adam state survives rebind | important | `reset_optimizer()` builtin + clear moments on model rebind | `7d89e953` |
| F21 adam in a user function | med | optimizer writes persist across the frame (train the real params) | `9d69cd04` |

Note (F22): Adam's step counter is per-optimizer (shared across params), so
`reset_optimizer()` is the full between-runs reset; rebinding a model clears
that model's moments.

- **F19** (BUG) -- a matmul inner-dimension mismatch inside `grad` PANICS the
  process ("compatible matmul shapes") instead of the structured shape error
  F18 gave elementwise ops. The F18 pre-validation covers the elementwise
  binary ops; the matmul tape op needs the same guard. Probe:
  `probes/f19_matmul_shape_panics_in_grad.mlpl`.
- **F20** (BUG) -- inside an inlined user function on the tape, `take`'s index
  parameter is not bound ("undefined variable"); worse, when a same-named
  global exists it silently resolves to THAT instead, and an out-of-range index
  then panics. Two problems: the F11/F15-class scope resolution (the index
  should resolve against the traced local scope) AND a panic on out-of-range
  that should be a clean error. Probe: `probes/f20_take_param_index_in_grad.mlpl`.
- **F21** -- `adam` called inside a user function trains function-LOCAL copies;
  the global models/params are unchanged afterward, with no error, so the loop's
  own evaluations look fine but nothing persists. Fix: resolve the optimizer's
  parameter list against the caller's bindings, or document the copy semantics
  loudly (error/warn). Probe: `probes/f21_adam_in_user_function.mlpl`.
- **F22** (important) -- `adam` keeps per-parameter state keyed by NAME, and
  that state survives rebinding the name to a new model, with no way to reset:
  training two variants in sequence under the same names silently trains the
  second with the first's moments. Fix: clear optimizer state when a name is
  rebound to a new model, and add a `reset_optimizer()` builtin. Probe:
  `probes/f22_adam_state_by_name.mlpl`.

## demo-coding-agent findings (2026-09-14, from ../demo-coding-agent) -- RESOLVED

A different downstream repo (a coding agent authoring/running MLPL) reported
five findings. Numbered CA1-CA5 here to avoid colliding with the moe-microscope
F1-F5 above; that repo files them as its own F1-F5. All shipped in the
`demo-coding-agent-findings` saga. Shared root: the array-centric error message
misled every guess at a natural builtin name -- CA2 (the meta-fix) surfaced the
rest.

| Finding | Resolution | Commit |
|---------|------------|--------|
| CA2 unknown-function diagnostic | undefined calls say "unknown function: NAME" | `c236f3dd` |
| CA5 `len` on string lists | polymorphic `len` (lists + arrays); `list_len` alias kept | `c66c24cd` |
| CA1 `"a" + "b"` | `+` concatenates two strings | `4eaa6b60` |
| CA4 `make_dir` | sandboxed directory creation (incl. parents) | `5dfc6d42` |
| CA3 symlink wording | docs corrected (behavior was already right) | `d92778db` |

Not yet built (design note only): `len_bytes`/`len_chars` for single-string
length -- `len("...")` currently errors and points at them.

- **CA2** (fix first) -- calling an undefined function reports the array
  diagnostic ("expected an array value, got a string") instead of "unknown
  function: NAME". This hid that `str_starts_with`/`str_trim` do not exist and
  misleads every wrong builtin-name guess. Fix: an undefined `name(...)` call
  errors with a clear unknown-function message.
- **CA1** -- `"a" + "b"` fails with the array diagnostic; string concatenation
  is `str_concat`/`str_join`. Fix or document: either make `+` on two strings
  concatenate, or give a clear "use str_concat" error. (Agents currently use
  str_concat/str_join.)
- **CA5** -- `len` rejects string lists; `list_len` is required. Fix:
  make `len` accept a StrList (item count), keeping `list_len` as an alias. See
  the design note on string length below.
- **CA4** -- `write_text` does not create parent directories, so an agent adding
  a module in a new directory is stuck. Fix: add a `make_dir` builtin (sandboxed
  like the other fs builtins), or have write_text create parents.
- **CA3** (doc-only) -- a symlink whose target is inside the sandbox reads fine,
  but the docs say symlinks are never followed. Behavior is good; fix the
  wording.

Design note (string length, CA5-adjacent): three notions exist -- bytes
(Rust `str::len`), code points (Python `len`, Rust `chars().count()`), and
grapheme clusters (neither counts by default). Recommendation: `len(StrList)` =
item count; a single-string `len` errors clearly and directs to `len_bytes`
(UTF-8 bytes) and `len_chars` (code points; document "not grapheme clusters").

## Follow-up batch 5 (2026-09-15 rerun by ../moe-microscope) -- RESOLVED

The rerun after `moe-microscope-followups-4` found F22 necessary-but-not-
sufficient and two more grad-surface scope findings. All shipped in
`moe-microscope-followups-5`.

| Finding | Severity | Resolution | Commit |
|---------|----------|------------|--------|
| F22 (redo) shared Adam step counter | important | step counter keyed per parameter (`adam:<param>`); `clear_param`/`reset_optimizer` drop it | `3ffd7266` |
| F23 reshape dims from fn params | BUG | reshape/windows/reduce/TensorCtor dims resolve through the traced scope; gradient flows | `fca6043c` |
| F24 apply_engram ids from fn param | BUG | ids resolve through the traced scope (same fix as gather_rows/F23); gradient reaches the memory table | `1b4d29e5` |

- **F22 (redo)** (important) -- `reset_optimizer()` plus rebind-clearing (batch
  4) were necessary but not sufficient: Adam's step counter was keyed
  per-optimizer (`"adam"`), shared across params, so two variants trained in
  sequence cross-contaminated through bias correction. Fix: key the step counter
  per parameter (`"adam:<param>"`) so each model's bias correction is
  independent; `clear_param`/`clear` drop it on rebind/reset. Probe:
  `probes/f22b_shared_step_counter.mlpl`.
- **F23** (BUG) -- a `reshape` whose dims are bound to a user function's
  arguments (not literals or globals) dropped the gradient inside `grad`: the
  dims were resolved with the eager global env, blind to the traced function-arg
  scope. Fix: thread the traced scope through `eval_shape_dims` (reshape dims,
  windows sizes/strides, reduce axes, TensorCtor shape) so a dim bound to a
  function argument resolves and the gradient flows. Probe:
  `probes/f23b_param_bound_reshape_in_grad.mlpl`.
- **F24** (BUG) -- `apply_engram`'s ids argument bound to a user-function
  parameter was resolved with the eager global env, so the gradient never
  reached the memory table (the same call with ids in a global traced fine).
  Fix: resolve ids through the traced scope (`eval_index_expr`), the identical
  fix already used by `gather_rows` and F23. Probe:
  `probes/f24_apply_engram_ids_param_in_grad.mlpl`.
