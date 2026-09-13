# Saga: moe-microscope-findings

Address the dogfooding findings the ../moe-microscope agent surfaced against
0.22.0 while building the MoE educational visualization demo (dogfooding
sw-mlpl + demo-extensions). Full work order with reproducers + triage:
`docs/sw-mlpl-findings.md`. Ordered by impact; F1 and F2 first.

## Steps

1. softmax-arity -- unify `softmax` so `softmax(x)` and `softmax(x, axis)`
   mean the same thing in BOTH eager evaluation and the grad/adam tape
   (default the last axis when omitted). Today eager requires the axis and
   the tape takes one arg, so a loss cannot be written once. TDD: the same
   `softmax(...)` expression evaluates AND trains; keep the explicit-axis
   form working.

2. userfns-in-grad -- support user-defined function calls (`u:name(...)`)
   inside `grad`/`adam` by tracing through the function body onto the tape,
   so a loss written as `def u:loss(...)` trains. The most important finding
   for "models as auditable source". TDD: `grad(u:loss(w), w)` matches the
   inline expansion; a gradcheck on a small user-fn loss.

3. chain-doc-or-fix -- confirm `chain(blk, blk, blk)` does not share weights
   (param_count triple-counts; adam "not a tracked parameter") and that
   nested `apply` is the intended weight-sharing/recurrence spelling. Fix the
   docs (glossary/lang-reference) to say so; only implement shared-weight
   `chain` if wanted. TDD/doc as appropriate.

4. gather-rows-backward-and-kl -- make `gather_rows` differentiable on the
   tape (scatter-add backward, like `windows`) so from-scratch addressing
   lessons can train; add a `kl_divergence` builtin (or document the
   softmax+log composition). TDD + gradcheck.

5. relay-and-close -- update `docs/sw-mlpl-findings.md` marking each finding
   resolved (or documented), relay to the moe-microscope agent, refresh
   CHANGES + wiki, mark the saga shipped in `docs/future-sagas-queue.md`.
   `--done`.
