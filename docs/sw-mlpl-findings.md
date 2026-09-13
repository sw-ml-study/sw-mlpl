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

## Follow-up batch 2 (2026-09-12 second rerun by ../moe-microscope) -- OPEN

A deeper dogfooding pass (the from-scratch TinyMoE "this domain" lesson: a
3,812-param block over a 120-example fixture) surfaced seven more findings,
each with a reproducer in the downstream `probes/`. F10 is a genuine bug (a
Rust panic); the rest are grad-surface gaps with workarounds. Queued as
`moe-microscope-followups-2`.

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
