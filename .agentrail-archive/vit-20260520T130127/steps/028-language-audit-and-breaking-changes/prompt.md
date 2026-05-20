Saga 29 step inserted: MLPL language analysis + breaking-change audit.

The user explicitly authorized breaking changes while MLPL is still in alpha: "We can make breaking changes to the language now but it will become harder to make breaking changes or major refactorings when we get to Beta or Release stages."

Produce docs/language-audit.md analyzing MLPL against established languages:

1. Comparisons. Cover at minimum:
   - APL / APL2 / J / BQN: tacit programming, forks/hooks/trains, primitives MLPL borrowed (rank, scan) vs. ones it lacks (rho/iota equivalents, etc.)
   - Python / PyTorch / JAX: ergonomics around shape checking, the closures-don't-differentiate problem, missing vmap/jit, batched/grad transforms vs MLPL's autograd tape
   - Rust: type discipline, traits-based dispatch, error types

2. Audit categories. For each issue, classify as:
   - MISSING (should be added)
   - INCONSISTENT (two builtins disagree on convention)
   - ERROR-PRONE (foot-gun that bites users)
   - ANTI-PATTERN (encouraged usage that scales badly)
   - ALPHA-LEAK (an implementation accident encoded into the surface)

3. Known issues to surface (seed list):
   - Train block requires inlining the forward expression because grad walks the AST, not let-bindings. Closures-don't-differentiate.
   - device("mlx") { } requires model params to be constructed INSIDE the scope or you get DeviceMismatch. Silent gotcha.
   - concat has overloaded arity (a, b) vs (a, b, axis) -- and the 3-arg form only supports axis in {0, 1}.
   - take is single-index; gather/slice ranges deferred.
   - Booleans encoded as 0.0/1.0 floats (no Bool type).
   - String literals for diagram types ("heatmap" vs "heatmap_grid") -- no namespaced types.
   - Magic seed constants threaded through every layer constructor.
   - sw-checklist budgets force odd code-shape choices (lots of submodule splits).
   - Various builtins have different naming conventions accreted across sagas.

4. For each issue, propose:
   - A specific breaking-change fix
   - The migration cost (demos / tests / docs to update)
   - Priority tier (critical / nice-to-have / cosmetic)

5. Output: docs/language-audit.md (long-form), plus a section in docs/plan.md "Breaking-change candidates" with the top-tier items pulled forward for saga consideration.

Quality gates: markdown-checker on the new doc; sw-checklist (should be unchanged since this is docs-only).