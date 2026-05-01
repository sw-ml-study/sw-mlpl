Saga 23 step 002: auto-tag-producers.

Wire the ops that *produce* typed values (Tier A vocabulary in docs/optional-typing-design.md) to attach the right ValueTag via Environment::set_tag at the assignment site, with no language surface (no annotation syntax, no :tag command -- those land in Saga 26).

Auto-tag rules to ship:
- softmax / sigmoid -> Probability
- log_softmax -> LogProbability
- linear / embed / attention constructors -> Weight{layer, name} for the W param, Bias{layer} for the b param
- tanh_layer / relu_layer / softmax_layer apply outputs -> Activation{layer, kind} (kind from existing ActKind via a small From impl)
- cross_entropy / mse / kl_divergence -> Loss{kind} with the matching LossKind variant
- grad(loss, w) -> Gradient{wrt: w-name}
- cosine_schedule / linear_warmup -> LearningRate
- attention_weights -> AttentionMap
- model apply final-layer output -> Logit (heuristic: outermost chain's last Linear; if followed by softmax/sigmoid, the softmax output gets Probability and the pre-softmax does not get Logit)

TDD: write failing tests in crates/mlpl-eval/tests/auto_tag_tests.rs covering each producer pair (e.g. y = softmax(x); env.get_tag("y") == Some(&ValueTag::Probability)). Then wire each producer in eval / runtime / model_dispatch to call Environment::set_tag at the right assignment-site hook.

Constraints:
- mlpl-eval is at the module budget; the auto-tagger logic spreads across existing producer modules, not a new module. Any helper that doesn't fit an existing module's 7-fn budget goes into mlpl-runtime or extracts a small new module under budget.
- Untyped values never disappear: a value with no auto-tag rule remains untagged.
- Zero new language surface; the only externally-visible change is that :describe (still untyped in this step) starts seeing tagged bindings -- but step 005 is what wires :describe to *show* tags. Step 002's externally-visible change is essentially internal until step 005 lands.
- Run full workspace cargo test to verify zero behavior change in existing demos.

Quality gates: /mw-cp full pass (test/clippy/fmt/sw-checklist). sw-checklist failed count must hold at 139 baseline.