Phase 1 step 002: `perturb_params` builtin + family pattern walker.

Add family-targeted Gaussian noise to a model's params in
place. This is the core of the Neural Thickets workflow.

1. New builtin `perturb_params(m, family, sigma, seed)` in the
   interpreter. Walks `m`'s params, filters by `family`, and
   for each matching param, adds `sigma * randn(seed, shape)`
   in place. Returns unit / the mutated model -- pick whichever
   matches the existing in-place-builtin convention and call
   it out in the contract.
2. Families (strings) -- exact set shipped in this step:
   - `all_layers`  -- every param.
   - `attention_only` -- name patterns `__attn_Wq_*`,
     `__attn_Wk_*`, `__attn_Wv_*`, `__attn_Wo_*`.
   - `mlp_only` -- `__linear_W_*` / `__linear_b_*` excluding
     the final projection head. Determine "final head"
     structurally: the last top-level `linear` child of the
     outermost `chain`. Do NOT use a name-only heuristic.
   - `embed_and_head` -- `__embed_E_*` plus the final `linear`'s
     `W` / `b`.
3. Unknown family strings raise `EvalError::InvalidArgument`
   (or the nearest existing error variant) with the accepted
   family list in the message. Numeric `sigma <= 0` passes
   through as "no-op or tiny"; do not special-case -- let
   `randn * sigma` behave naturally.
4. Contract file `contracts/eval/perturb-params.md` with:
   input types, family list + patterns, structural head
   definition, in-place semantics, error cases, seeding rule
   (document that two calls with the same seed produce
   identical deltas).
5. TDD (RED -> GREEN -> REFACTOR) in
   `crates/mlpl-eval/tests/perturb_params_tests.rs`. Use a
   fixture model big enough to contain every family:
   `chain(embed(V, d, 0), residual(chain(rms_norm(d),
   causal_attention(d, 1, 1))), residual(chain(rms_norm(d),
   linear(d, 4*d, 2), relu_layer(), linear(4*d, d, 3))),
   rms_norm(d), linear(d, V, 4))`.
   - Clone base (step 001); `perturb_params(clone,
     "attention_only", 0.02, 42)`; attention params differ
     from base within `|delta| in [0, 6*sigma]`, MLP + embed
     + head params bit-identical.
   - Same but `"mlp_only"`, `"embed_and_head"`,
     `"all_layers"`; spot-check that each family touches
     exactly the intended subset and nothing else.
   - Determinism: two clones perturbed with same seed match
     bit-for-bit; different seeds differ.
   - Unknown family "banana" raises the expected error with
     the family list in the message.
6. `mlpl-rt` parity: same judgement call as step 001 -- port
   only if needed for existing parity tests; otherwise note
   in the contract.
7. Quality gates + `/mw-cp`. Commit message references
   Saga 20 step 002.
