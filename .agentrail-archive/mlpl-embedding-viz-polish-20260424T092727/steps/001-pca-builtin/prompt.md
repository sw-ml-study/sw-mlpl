Phase 1 step 001: `pca(X, k) -> Y [N, k]` builtin.

Top-k Principal Component Analysis via power iteration
+ deflation, wrapped into a single builtin so callers
can replace the column-selector or hand-rolled
power-iteration idioms with one line.

1. **Signature**.
   - `X` rank-2 `[N, D]`.
   - `k` positive integer scalar, `1 <= k <= D`.
   - Returns rank-2 `[N, k]` of centered-and-
     projected data (the principal components
     themselves are NOT returned as a separate
     output; single-return keeps the builtin
     surface simple).

2. **Algorithm**.
   a. Center: `Xc[i, j] = X[i, j] - col_mean_j`.
   b. Covariance: `Cov = Xc^T @ Xc / N`, shape
      `[D, D]`.
   c. For each of the `k` components:
      - Initialize `v = [1, 0, ..., 0]` (or seeded;
        document choice).
      - Power-iterate for 50 iterations:
        `v = Cov @ v; v = v / ||v||`.
      - Extract eigenvalue `lambda = v^T @ Cov @ v`.
      - Store component `v` as row `i` of `V [k, D]`.
      - Deflate: `Cov = Cov - lambda * (v @ v^T)`
        so the next iteration finds the next-
        largest-variance direction.
   d. Project: `Y = Xc @ V^T`, shape `[N, k]`.
   e. Return `Y`.

3. **Module**: new
   `crates/mlpl-runtime/src/pca_builtin.rs`.
   Small enough to stay in one module -- orchestrator
   plus 2-3 pure helpers (`center_data`, `compute_cov`,
   `extract_components`). Design for the sw-checklist
   budgets up front per
   `docs/sw-checklist-patterns.md`.

4. **Contract** at
   `contracts/eval-contract/pca.md`:
   - Signature, algorithm, hyperparameter choices
     (50 power iterations, deterministic init).
   - Rotation/reflection ambiguity: eigenvectors can
     flip sign; absolute signs of Y columns are not
     guaranteed stable across numerical perturbations.
   - Error cases: rank != 2, k == 0, k > D, non-
     finite X.
   - Non-goals: no sparse PCA, no SVD-based PCA
     (power iteration is simpler and fine at our
     scale), no components returned separately.

5. **TDD** (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-eval/tests/pca_tests.rs`:
   - Anisotropic 2-D fixture:
     `X = matmul(randn(0, [60, 2]), [[1, 0.3], [0, 0.1]])`
     stretches heavily along one axis. Run
     `pca(X, 1)`; assert the returned `[60, 1]`
     captures >80% of total variance (one-line
     variance comparison against the original).
   - Shape: `[N, D]` + `k` -> `[N, k]` for multiple
     (N, D, k) combinations.
   - `k == D`: the projection preserves total
     variance within a 1e-6 numerical tolerance.
   - Determinism: two calls with identical input
     produce bit-identical output.
   - Error paths: rank-1 input, rank-3 input,
     `k = 0`, `k > D`, non-finite entries, wrong
     arity.

6. Quality gates + `/mw-cp`. Commit message
   references Saga 16.5 step 001.
