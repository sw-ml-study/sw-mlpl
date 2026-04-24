Phase 1 step 001: `pairwise_sqdist(X)` + `knn(X, k)`
builtins.

Two sibling builtins that together cover the
"distance-based inspection" surface for an embedding
table (or any rank-2 array viewed as a set of points).

1. **`pairwise_sqdist(X) -> D`**.
   - `X` is rank-2 `[N, D]`.
   - `D` is rank-2 `[N, N]` where
     `D[i, j] = sum over k of (X[i, k] - X[j, k])^2`.
   - Implementation should compose the
     `matmul + reduce_add + broadcasting` identity so it
     inherits MLX dispatch for free. (The K-Means demo
     in Saga 8 uses this same identity inline; Saga 16
     promotes it to a builtin so the shape check + the
     3-line boilerplate live in one place.)
   - Shape check: rank != 2 -> error; N == 0 ->
     returns empty `[0, 0]` gracefully (do not panic).

2. **`knn(X, k) -> idx [N, k]`**.
   - `X` is rank-2 `[N, D]`; `k` is a positive integer
     scalar with `k < N`.
   - Returns `[N, k]` of integer-valued `f64` indices.
     Row `idx[i]` holds the `k` nearest neighbors of
     `X[i]` by squared Euclidean distance, sorted by
     ascending distance.
   - **Self-exclusion**: `i` never appears in `idx[i]`
     (point itself is not its own neighbor).
   - **Tie-break**: lower index wins; stable sort per
     row.
   - Errors: rank != 2, `k <= 0`, `k >= N` (need at
     least one non-self neighbor per row), non-integer
     k.

3. Module placement: new
   `crates/mlpl-runtime/src/embedding_builtins.rs` (small
   module, under the 7-fn budget). Dispatched from the
   existing `call_builtin` chain in
   `crates/mlpl-runtime/src/builtins.rs` following the
   `ensemble_builtins` / `random_builtins` pattern.

4. Contracts:
   - `contracts/eval-contract/pairwise-sqdist.md`
   - `contracts/eval-contract/knn.md`

5. TDD (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-eval/tests/pairwise_sqdist_knn_tests.rs`:
   - `pairwise_sqdist` of a hand-constructed 3-point
     fixture: symmetry (`D[i,j] == D[j,i]`), zero
     diagonal, specific computed values.
   - `knn(X, 1)` for 4 points on a 1-D line: each picks
     its nearest non-self neighbor (2 ties resolved by
     lower index).
   - `knn(X, 2)` returns indices sorted by ascending
     distance.
   - Self-exclusion: assert `i` never appears in
     `idx[i]` for any i.
   - Error paths: rank-3 input, k=0, k>=N, non-integer
     k (surface `RuntimeError::InvalidArgument`).

6. Quality gates + `/mw-cp`. Commit message references
   Saga 16 step 001.
