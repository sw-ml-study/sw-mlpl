Phase 1 step 001: `clone_model` builtin.

Deep-copy a `ModelSpec` tree so perturbation can mutate a
copy without touching the base. This is the foundation for
every other Saga 20 step.

1. New builtin `clone_model(m) -> Model` in the interpreter:
   walks the `ModelSpec` tree, allocates fresh param names
   (stable rename scheme, e.g. suffix with a fresh clone id
   so repeated clones stay distinct), copies the stored
   values, returns an independent `ModelSpec`.
2. Contract file `contracts/eval/clone-model.md` describing:
   input/output types, param-name rename scheme, independence
   guarantee (mutating the clone's params does not change the
   base), identity guarantee (forward output equals base's on
   the same input before any perturbation).
3. TDD (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-eval/tests/clone_model_tests.rs`:
   - Construct `chain(linear(4, 4, 0), linear(4, 4, 1))` base.
   - `clone_model(base)` forward-equals base on a fixed input
     (bit-identical or within documented fp32 tolerance).
   - Mutate the clone's params (train it one step via `adam`,
     or directly via `perturb_params` once step 002 lands --
     for step 001 use the one-step-adam route) and assert the
     base's params are unchanged.
   - `clone_model(clone_model(base))` works and the two clones
     have distinct param names from each other and from base.
4. `mlpl-rt` parity: if the compile-to-Rust path needs the
   builtin to keep parity tests green, port it there too. If
   the compiled path is not exercising `clone_model` yet, note
   that in the contract and leave a follow-up rather than
   over-building.
5. Quality gates + `/mw-cp`. Commit message references
   Saga 20 step 001.
