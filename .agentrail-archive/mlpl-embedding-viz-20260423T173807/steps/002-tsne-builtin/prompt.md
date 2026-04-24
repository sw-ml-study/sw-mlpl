Phase 2 step 002: `tsne(X, perplexity, iters, seed)`
builtin.

Classic van der Maaten t-SNE: perplexity-calibrated
conditional probabilities in the high-dim space,
Student's-t affinities in the low-dim space, KL
divergence loss, gradient descent. Output is 2-D.

1. **Signature**.
   - `tsne(X, perplexity, iters, seed) -> Y`
   - `X` rank-2 `[N, D]` (the points to embed).
   - `perplexity` positive f64, typically 5-50; must
     satisfy `perplexity < (N - 1) / 3` per van der
     Maaten's guidance.
   - `iters` positive integer number of gradient
     descent steps.
   - `seed` f64 for the initial Y init
     (`randn(seed, [N, 2]) * 1e-4`).
   - Returns `Y` rank-2 `[N, 2]`.

2. **Algorithm** (document each hyperparameter in the
   contract):
   - Compute pairwise squared distances `D2` via
     `pairwise_sqdist(X)` (step 001).
   - Per-row binary search for `beta_i` such that
     the Shannon entropy of `P_{j|i} = exp(-beta_i *
     D2[i, j]) / Z_i` matches `log(perplexity)`. 50
     bisection iterations or until entropy diff < 1e-5.
   - Symmetrize: `P_ij = (P_{j|i} + P_{i|j}) / (2 * N)`;
     clamp to `max(P_ij, 1e-12)`.
   - Low-dim affinities: `Q_ij = 1 / (1 + ||Y_i -
     Y_j||^2)` normalized by their sum.
   - Gradient: `dY_i = 4 * sum_j (P_ij - Q_ij) * Q_ij_unnorm
     * (Y_i - Y_j)`.
   - Update with learning rate 200, momentum 0.5 for
     the first 250 iters then 0.8, early exaggeration
     factor 4 for the first 100 iters.
   - Center Y at each step so the solution does not
     drift.

3. **Implementation**: pure-Rust builtin in
   `crates/mlpl-runtime/src/embedding_builtins.rs`. Do
   NOT attempt to express the inner loop through the
   autograd tape -- the per-point perplexity calibration
   is a binary search that does not vectorise cleanly
   and is cheaper to ship as a direct kernel. The
   `pairwise_sqdist` call at the start reuses step 001's
   builtin so it inherits MLX dispatch automatically if
   the caller is inside `device("mlx") { }`.

4. Contract `contracts/eval-contract/tsne.md`: document
   every hyperparameter choice + its source + the
   rotational / reflection ambiguity in the output + the
   seed-sensitivity of the result.

5. TDD (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-eval/tests/tsne_tests.rs`:
   - **Structure-preservation**: three well-separated
     Gaussian blobs in `[60, 4]`
     (use `randn` + a fixed center-offset pattern).
     Run `tsne(X, 15.0, 200, 42)`. Verify each cluster's
     2-D centroid is pairwise-distinct (min pairwise
     distance > threshold). Does not assert exact
     coordinates (t-SNE has rotational + reflection
     symmetry and is seed-sensitive).
   - **Shape**: input `[N, D]` -> output `[N, 2]` for
     any D.
   - **Determinism**: two calls with identical
     `(X, perplexity, iters, seed)` produce
     bit-identical Y.
   - **Error paths**: rank != 2 X, perplexity <= 0,
     perplexity >= (N-1)/3 (the binary search would not
     converge), iters < 1, non-finite X entries.

6. Quality gates + `/mw-cp`. Commit message references
   Saga 16 step 002.
