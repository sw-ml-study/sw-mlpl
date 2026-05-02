Saga 23 step 005: tag-propagation.

Define and implement a propagation table for ValueTags through MLPL ops so a tag applied to an input flows through arithmetic, transpose, reshape, reductions to the result. This is the contract that makes tags compose -- without it, every op except the named producers strips the tag.

Propagation rules to ship:

1. Arithmetic (eval_ops.rs eval_binop):
   - Logit + Logit -> Logit (combining unnormalized scores stays unnormalized).
   - Loss + Loss -> Loss (sum of losses is still a loss; kind: from lhs).
   - Tagged + Untagged or Untagged + Tagged -> the tagged side wins.
   - Logit + Probability or Probability + anything -> error with hint:
     'Logit and Probability live in different domains. combine via cross_entropy or convert with softmax / log.'
   - Two different tag families (e.g. Loss + Probability) -> error with similar hint.

2. transpose / reshape_labeled: preserve tag (semantic identity intact under axis permutation).

3. reshape: clear tag (shape reflow loses semantic identity).

4. reduce_add / reduce_mul / mean / argmax over a tagged value:
   - Loss survives reduction (a partial sum of losses is still a loss).
   - Probability does NOT survive reduction (a partial sum of probs is not a probability) -- clear.
   - Default: clear unless explicitly survivable.

5. Negation (UnaryNeg): preserve tag (sign flip doesn't change domain).

Implementation:
- New helper module crate::tag_propagate (or extend auto_tag.rs if under 7-fn budget).
- propagate_arith(lhs: Option<&ValueTag>, rhs: Option<&ValueTag>, op: BinOpKind) -> Result<Option<ValueTag>, EvalError> -- runs at the assignment site for tagged LHS-or-RHS.
- propagate_transpose / propagate_reshape / propagate_reduce -- per-op helpers.
- The current Expr::Assign hook in eval.rs already calls auto_tag::for_assign; add a fallback for binops/transpose/reduces that consults tag_propagate.

Constraints:
- The auto_tag.rs producer rules in steps 002/003 take precedence -- if for_assign returns Some(tag), use it (the producer is authoritative). Propagation only fires when for_assign returns None and at least one input is tagged.
- Errors raised here use EvalError::TypeMismatch with the same op/expected/actual/hint shape as step 004.
- Ship a contract doc at contracts/eval-contract/tag-propagation.md describing the rules in prose.

TDD: failing tests in crates/mlpl-eval/tests/tag_propagation_tests.rs covering each rule.

Quality gates: full /mw-cp pass. sw-checklist failed count must hold at 139.