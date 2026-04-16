Phase 2 step 004: device("...") scoped form + parser surface.

Expose device placement in the MLPL source language.

1. Parser: new `Expr::Device { target: String, body:
   Box<Expr> }` AST node for `device("mlx") { body }` and
   `device("cpu") { body }`. Grammar mirrors `experiment
   "name" { body }`. Round-trip test: source -> parse ->
   pretty-print matches.
2. Evaluator: inside a `device("mlx") { ... }` block, array
   allocations and ops dispatch through `mlpl-mlx` when the
   `mlx` feature is compiled in. On non-MLX hosts, emit a
   one-time warning and fall back to the CPU runtime.
   `device("cpu") { ... }` is always a no-op (same values,
   same labels, same shapes) and must work on every host.
3. Nesting: a `device` block inside another `device` block
   uses the inner target. Test both directions of nesting
   with `experiment { device { ... } }` and `device {
   experiment { ... } }`.
4. Labels and shapes propagate across the block boundary
   unchanged -- a `[batch, feat]` label on a tensor allocated
   inside `device("mlx") { }` is still `[batch, feat]` when
   read outside.
5. TDD: parser test (red -> green -> refactor), then
   evaluator tests.
6. Quality gates + `/mw-cp`. Commit message references step 004.
