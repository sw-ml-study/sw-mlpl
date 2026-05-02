Saga 23 step 004: predicate-consumers.

Wire the ops that *consume* typed values to assert preconditions on entry. Each predicate failure raises a new EvalError::TypeMismatch { op, expected, actual, hint } variant with a tutoring hint pointing at the most likely fix. This is the saga's correctness payoff: cross_entropy(softmax(logits), Y) becomes a TypeMismatch with a 3-5 line tutor instead of a NaN factory.

Predicate rules to ship:

- cross_entropy(logits, y) -- arg 0 expected Logit; reject Probability with hint:
    'cross_entropy expects raw scores (Logit), not probabilities. you applied softmax already (line N). use one of:
       cross_entropy(logits, y)     -- pass the original Logit
       nll(probs, y)                -- if you already have probabilities'
  Also reject LogProbability with a similar hint pointing at nll.

- sample(logits, t, seed) -- arg 0 expected Logit; reject Probability/LogProbability with hint
  about double-sampling.

- top_k(logits, k) -- arg 0 expected Logit.

- entropy(probs) -- arg 0 expected Probability or LogProbability (Saga 24 will add proper
  Distribution support; for now this exists as a follow-up if a builtin lands).

- adam(loss, params, ...) / momentum_sgd(loss, params, ...) -- arg 0 expected Loss; reject
  Probability/Logit/etc with hint pointing at cross_entropy/mse.

- confusion_matrix(preds, labels) -- not a typed predicate yet (would need Labels{num_classes}
  match between args); deferred to step 005 when Labels gets auto-tagged from one_hot inputs.

- nll -- not a current builtin; defer until Saga 24 distribution work surfaces it.

Untyped values pass every predicate -- the gradual-typing additivity rule. A user passing an
untagged Probability-shape array to cross_entropy still gets the existing shape-based behavior.

TDD: failing tests in crates/mlpl-eval/tests/predicate_consumer_tests.rs covering each pair:
positive cases (Logit -> cross_entropy works), negative cases (Probability -> cross_entropy
TypeMismatch with hint substring match), untyped passes, etc.

Implementation:
- New EvalError::TypeMismatch variant in crates/mlpl-eval/src/error.rs.
- New crates/mlpl-eval/src/type_errors.rs module (under 7-fn budget) holding the tutoring hint
  catalog as a small lookup (op-name + got-tag -> hint string).
- Hook the predicate check into eval_fncall (or a new wrapper) BEFORE dispatched_call. Look up
  arg names if they are Idents, fetch tags, run the predicate, raise TypeMismatch if it fails.

Quality gates: full /mw-cp pass. sw-checklist failed count must hold at 139.