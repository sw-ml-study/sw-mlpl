Phase 2 step 004: Cross-entropy loss over integer targets.

Add a numerically-stable, fully-differentiable cross-entropy
loss so the LM can be trained with `adam` in a `train { }`
loop without hand-rolling log-softmax + NLL.

1. New builtin `cross_entropy(logits, targets)`:
   - `logits`: `[N, V]` float (also accept `[B, T, V]` by
     reshaping internally to `[B*T, V]` and reshaping back the
     scalar -- the result is always a scalar mean).
   - `targets`: `[N]` (or `[B, T]`) integer-typed array.
   - Returns scalar `-mean(log_softmax(logits)[i,
     targets[i]])`.
2. Implementation: log-softmax via max-subtraction for
   stability:
       lse = max(z) + log(sum(exp(z - max(z))))
       log_p = z - lse
       loss = -mean(gather(log_p, targets))
   All ops must be tape primitives so `grad(...)` works.
3. Errors:
   - Shape mismatch produces `EvalError::ShapeMismatch { op:
     "cross_entropy", expected, actual }`.
   - Out-of-range target indices return a clear
     `EvalError::Other` (or a new variant) -- do not panic.
4. TDD:
   - Numeric: `[4, 3]` example with hand-computed expected
     loss matches within 1e-6.
   - Gradcheck against finite differences for `d loss / d
     logits` on a `[3, 4]` case.
   - Stability: very large logits (1e3+) do not produce inf /
     nan.
   - Wrong-shape targets surface a clean error message.
5. Quality gates + `/mw-cp`. Commit message references step 004.
