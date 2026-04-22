Phase 1 step 003: `argtop_k` + `scatter` builtins.

Close the ensemble loop with two small utility builtins.

1. `argtop_k(values, k) -> IntArray[k]`. Returns the k indices
   of the largest entries in a rank-1 `values` array.
   - Tie-break rule: lower index wins (document and test).
   - `k > len(values)` raises
     `EvalError::InvalidArgument`.
   - Returned indices are NOT guaranteed sorted by index; they
     are sorted by descending value. Document this and test it.
   - Distinct name from existing `top_k(logits, k)` (which
     masks logits) because the return type differs.

2. `scatter(buffer, index, value) -> Array`. Returns `buffer`
   with `buffer[index] = value`. Rank-1 buffer, scalar integer
   `index`, scalar `value`.
   - Out-of-range `index` raises
     `EvalError::IndexOutOfBounds`.
   - Non-rank-1 `buffer` raises the shape error.
   - Semantics are "return new array" at the MLPL source level
     even if the runtime mutates in place -- consistent with
     how other array ops present themselves.

3. Contract files:
   - `contracts/eval/argtop_k.md`
   - `contracts/eval/scatter.md`

4. TDD (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-eval/tests/argtop_k_tests.rs` and
   `crates/mlpl-eval/tests/scatter_tests.rs`:
   - `argtop_k([0.1, 0.5, 0.2, 0.9], 2) == [3, 1]`.
   - `argtop_k([1.0, 1.0, 0.0], 2) == [0, 1]` (tie -> lower
     index).
   - `argtop_k([...], 5)` with len 3 -> error.
   - `scatter(zeros([4]), 2, 7.5) == [0, 0, 7.5, 0]`.
   - `scatter(zeros([4]), -1, 1.0)` and `scatter(zeros([4]),
     4, 1.0)` -> error.
   - `scatter` on a non-rank-1 array -> shape error.

5. `mlpl-rt` parity: same judgement call as prior steps --
   port only if parity tests demand it. The Saga 20 demo runs
   through the interpreter.

6. Quality gates + `/mw-cp`. Commit message references
   Saga 20 step 003.
