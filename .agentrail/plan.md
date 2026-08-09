# Saga: compiler-apply-binop

Fix the compile-to-Rust arithmetic defect: lowered code calls
.apply_binop (the ApplyBinopExt trait) but the trait is not in
scope in generated Rust, so documented arithmetic compilation
fails. Root cause + fix:

- mlpl-rt does not re-export ApplyBinopExt -> re-export it.
- lower-rs emits a bare method call -> emit a UFCS call
  (#rt::ApplyBinopExt::apply_binop(&l, &r, closure)) so no use is
  needed and no unused-import warning arises for binop-free
  programs.
- The two gated compile tests (mlpl-lower-rs compile_tests,
  mlpl-macro macro_compile_tests) have STALE workspace paths
  (crates/mlpl-rt / crates/mlpl) from before the cellular
  monorepo, so they cannot run and never caught this. Fix the
  path computation so they are valid regressions.

## Steps
1. fix -- re-export ApplyBinopExt; UFCS lowering; fix gated-test
   workspace paths; run the gated compile tests green.
2. close -- note arithmetic compiles in status/maturity; --done.
