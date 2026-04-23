Phase 1 step 002: `lora(m, rank, alpha, seed)` builtin +
`ModelSpec::LinearLora` variant.

Replace every `Linear` node in a model's spec tree with a
LoRA-wrapped variant that owns two fresh adapter matrices
alongside the base W, b.

1. New `ModelSpec::LinearLora { w, b, a, b_adapter,
   in_dim, out_dim, rank, alpha }` variant in
   `crates/mlpl-eval/src/model.rs`.
   - `w`, `b` are the base-weight param names (reused from
     the original `Linear`).
   - `a`, `b_adapter` are the fresh adapter param names
     allocated at `lora()` call time (use names like
     `__lora_A_{id}` and `__lora_B_{id}` with fresh ids
     from `env.next_model_id`).
   - `in_dim`, `out_dim`, `rank`, `alpha` are the
     structural constants (also cached for shape checks
     and the forward formula).
   - Extend `ModelSpec::params()` to return all four
     names.
2. New `lora(m, rank, alpha, seed)` builtin in a new
   `crates/mlpl-eval/src/model_lora.rs` module (parallel
   placement to `model_clone.rs`, `model_perturb.rs`). It:
   a. Clones `m`'s spec tree via the `clone_model` path
      (step 001 of Saga 20), producing a fresh-named
      `ModelSpec` tree whose base param values match `m`.
   b. Walks the cloned tree and replaces every
      `ModelSpec::Linear { w, b }` with a
      `ModelSpec::LinearLora { w, b, a, b_adapter,
      in_dim, out_dim, rank, alpha }`.
   c. Allocates `a_name = format!("__lora_A_{id}")` at
      shape `[in_dim, rank]` with values
      `randn(seed + i, [in_dim, rank]) * (1.0 / sqrt(in_dim))`
      (standard LoRA A init; `i` is a running index over
      replaced Linears so same-shape Linears get
      independent deltas).
   d. Allocates `b_adapter_name = format!("__lora_B_{id}")`
      at shape `[rank, out_dim]` with values all zero. The
      zero-init on B is the LoRA-standard
      "pre-training-step delta is zero" property:
      `apply(lora_m, X) == apply(m, X)` before any training
      step.
   e. Registers both adapter names via `env.set_param(...)`
      and propagates the device tag from the cloned `w`
      via `env.set_tensor_device(...)` (same pattern as
      `clone_model`).
3. Accept the same identifier-or-expression argument shape
   for `m` as `clone_model` / `perturb_params`.
4. Error cases:
   - Wrong arity (not 4 args) -> `BadArity`.
   - Non-model `m` -> `Unsupported("lora: ... not a model")`.
   - `rank <= 0` -> `Unsupported("lora: rank must be positive, got ...")`.
   - `rank > min(in_dim, out_dim)` on ANY replaced Linear
     -> `Unsupported("lora: rank R exceeds min(in=IN, out=OUT) for layer X")`.
     Surface the offending layer's structural context in
     the message so the user knows which Linear is the
     problem.
5. Contract file `contracts/eval-contract/lora.md`:
   - Signature, param-name convention, init convention
     (including the zero-init-B rationale), device-tag
     propagation, error cases, non-goals (no selective
     attachment, no adapter merging, no quantization;
     all deferred).
6. TDD (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-eval/tests/lora_tests.rs`:
   - `m = linear(4, 8, 0); lora_m = lora(m, 2, 4.0, 7)`:
     param names include W/b from clone + A/B adapter;
     shapes are `[4, 8]`, `[1, 8]`, `[4, 2]`, `[2, 8]`.
   - Zero-init on B: `env.get(b_adapter_name).data()` is
     all zeros.
   - Scaled-randn on A: values differ from zero and match
     `randn(seed, ...) * (1 / sqrt(in_dim))` within
     floating-point tolerance.
   - Forward identity: `apply(lora_m, X) == apply(m, X)`
     (elementwise) before any training -- i.e. zero-init
     B makes the adapter a no-op initially.
   - Chain with multiple linears: `lora(chain(linear(...),
     relu_layer(), linear(...)), 2, 1.0, 0)` wraps both
     linears; non-linear nodes unchanged.
   - Nested embed / rms_norm / residual / attention:
     lora leaves those untouched (Saga 15 is Linear-only).
   - Error paths: arity, non-model, rank <= 0, rank too
     large.
7. Module placement: `crates/mlpl-eval/src/model_lora.rs`
   (new). Wire `mod model_lora;` in `lib.rs` and add the
   constructor-dispatch branch in `eval.rs` next to the
   existing `clone_model` entry.
8. Quality gates + `/mw-cp`. Commit message references
   Saga 15 step 002.
