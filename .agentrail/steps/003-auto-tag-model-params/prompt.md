Saga 23 step 003: auto-tag-model-params.

Continue the producer auto-tagging from step 002 with the three model-aware rule clusters that step 002 deferred. These need structural inspection of ModelSpec / hooks into model_dispatch, not just AST matching at the assign site.

Rules to ship:

1. Weight / Bias tags on param creation by model constructors:
   - linear(in, out, seed) -> Weight{layer: <model_id>, name: "W"} for the W param,
     Bias{layer: <model_id>} for the b param (created in model_dispatch.rs around
     line 46-50 where set_param fires).
   - embed(vocab, d_model, seed) -> Weight{layer, name: "table"} for the lookup table.
   - attention(d_model, heads, seed) -> Weight{layer, name: "W_q"|"W_k"|"W_v"|"W_o"} for
     the four projection matrices.
   - The <layer> string is the model_id used in param naming. Pull from env.next_model_id
     bookkeeping at construction time.

2. Logit on apply(model, X) result when the model's structural tail is a Linear:
   - Walk ModelSpec to find the outermost-chain's last child.
   - If it is a Linear, the apply result auto-tags Logit.
   - If it is followed by softmax_layer or sigmoid as the final activation, the apply
     result auto-tags Probability instead (and the pre-softmax intermediate is
     anonymous, no tag).
   - If the final layer is an arbitrary activation (tanh_layer / relu_layer), tag is
     Activation{layer: model_id, kind: matching ActivationKind}.
   - Helper module crate::model_tag (new module) walks ModelSpec; mlpl-eval is over the
     module-count FAIL anyway so adding one more is a no-op for that gate. Keep its
     fn count under 7.

TDD: failing tests in crates/mlpl-eval/tests/auto_tag_models_tests.rs covering each
constructor (Weight/Bias on param names) and each apply-tail case (Logit, Probability,
Activation). Then implement.

Constraints:
- Param tagging happens AT param creation (model_dispatch.rs), so set_tag is called
  with the param name (e.g. "linear_3_W") rather than the binding name. Tags follow
  param names just like is_param/is_frozen do.
- apply-result tagging happens at the assign site (Expr::Assign), but the rule needs
  to look up the model's ModelSpec via env.get_model. Add a model-aware helper to
  auto_tag and dispatch from for_assign when value is FnCall { name: "apply", .. }.
- Untyped models stay untyped: a model without a clean structural tail just gets no
  tag on its apply result.
- Run full workspace cargo test to verify zero behavior change.

Quality gates: full /mw-cp pass. sw-checklist failed count must hold at 139.