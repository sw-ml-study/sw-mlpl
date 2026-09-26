# grad-soundness-records

Four grad fixes reported by ../demo-decision-model (its finding Q5, reproducer
`probes/q5_record_field_in_grad.mlpl`), all in `mlpl-eval`'s grad
mini-evaluator plus one new autograd node. Ordered by severity: a silent
wrong-gradient bug first, then two usability fixes, with the perf node between
them because it unblocks full-batch training downstream.

## Findings (verified 2026-09-22 against target/release/mlpl-repl)

- **Fold soundness bug.** `grad_const::fncall_or_fold` falls back to eager
  constant-folding when tape tracing of a call fails. Its purity check
  `differentiably_uses_param` inspects only the call's ARGUMENTS, not a `u:`
  function's BODY, so `u:f(b)` whose body reads global param `W` is folded to a
  constant. Result: `grad(u:f(b), W)` blames the param ("the loss does not
  depend on 'W'"), and `grad(u:f(b) + reduce_add(W), W)` returns all 1s -- a
  silently WRONG gradient (the u:f term is dropped). `eval_grads_batch` then
  zero-fills the missing grad, so optimizer steps silently do nothing.
- **shape() inside a u: function.** F12 exempts shape/rank/len/labels in
  `differentiably_uses_param`, but `fold_const_expr` evaluates in the global env
  without the traced local overlay, so `shape(a)` on a function parameter fails:
  "grad: function 'shape' not supported inside grad()".
- **gather_rows is dense.** `grad_calls_engram::call_gather_rows` builds a
  one-hot `[n, V]` selection matrix and matmuls it with the table: O(n*V*D)
  forward AND backward, n*V*8 bytes. ~30 s/step at n = 7595*24. The embedding
  layer in `mlpl-models-tape/src/layers.rs` uses the same one-hot pattern.
- **Record field reads.** `Expr::FieldAccess` is not a supported form in
  `eval_tensor_expr`, and the traced scope (`HashMap<String, Tensor>`) cannot
  bind a record-valued `u:` argument.

## Steps

1. fold-soundness -- make `differentiably_uses_param` conservative for `u:`
   calls (treat as param-using, or walk the body's free idents against
   params), so the real tracing error surfaces instead of a fold. Improve the
   unsupported-form message to name the form (e.g. "record field access `r.ids`
   inside grad"). Decide + document whether a train step where a param gets no
   gradient should warn instead of silently zero-filling. TDD: the
   `u:f(b) + reduce_add(W)` case must ERROR (not return 1s); the one-deep
   record read must name the field access, not the param.
2. shape-metadata -- `shape`/`rank`/`len` inside grad evaluate their traced
   argument's forward value and return a constant leaf (add to the
   stop-gradient path or overlay locals in `fold_const_expr`). TDD:
   `def u:g(a) { reduce_add(a) * take(shape(a),0,0) }`, `grad(u:g(W), W)`.
3. gather-node -- native `GatherRows` tape node in mlpl-autograd-tape: forward
   copies rows, backward scatter-ADDs upstream into only the touched rows
   (duplicates accumulate), O(n*D). Rewire `call_gather_rows` and the
   models-tape embedding one-hot path onto it. Resident/mlx path: host fallback
   is acceptable if no device kernel exists -- note it. TDD: gradcheck vs the
   old one-hot result incl. duplicate ids; a large-vocab timing sanity test.
4. record-fields -- record field reads inside grad as CONSTANT leaves: records
   are never params (param identity is by name), so a field read is data.
   Extend the traced scope with a side map for non-tensor constants so a
   record-valued `u:` argument binds; `FieldAccess` on it yields an untracked
   leaf. Records of weights stay non-trainable -- document that. TDD: the Q5
   probe written with a `{ids, wmask}` record matches the plain-array gradient.
5. relay-close -- docs (lang-reference grad section, sw-mlpl-findings), answer
   demo-decision-model in docs/q-and-a.md (their 12-arg workaround can go),
   future-sagas-queue, CHANGES, wiki errata if claims changed. Rebuild
   mlpl-repl release+debug. --done.
