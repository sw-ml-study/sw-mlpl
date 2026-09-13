# sw-mlpl dogfooding findings (from ../moe-microscope)

Status: WORK ORDER (planning doc). Findings surfaced by the moe-microscope
repo agent probing mlpl-repl 0.22.0 while building the MoE educational
visualization demo (dogfooding sw-mlpl + demo-extensions, not just proving a
model trains). Each has a reproducer; fixes are a SEPARATELY AUTHORIZED saga
(see docs/future-sagas-queue.md). Reproduced here against 0.22.0 unless noted.

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

## F3 -- chain(blk, blk, blk) does not share weights (MEDIUM; workaround exists)

`chain` triple-counts `param_count` and `adam` fails with "not a tracked
parameter". Nested `apply(blk, apply(blk, apply(blk, x)))` IS the working
spelling for weight-shared recurrence.

Proposed fix: either make `chain` share the block's weights (single tracked
parameter set) or document clearly that `chain` does NOT share and nested
`apply` is the recurrence/weight-sharing spelling.

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

## Triage

- F1, F2 are the strong candidates for a dedicated, authorized sw-mlpl saga --
  they block writing a model as reusable source and reusing one loss
  expression. F2 is the most important for the "models as auditable source"
  goal.
- F3 is likely documentation (nested `apply` is the intended spelling); confirm
  and document, only implement `chain` sharing if wanted.
- F4 is a small additive pair (one tape kernel + one composition builtin).

None blocks the moe-microscope Saga 1 or 2; each is a ledger entry with a
reproducer.
