Saga 33 step 007: split model_dispatch.rs (bonus / continuation).

model_dispatch.rs is the third big carry-over from saga 32: 905 lines (File-LOC FAIL), 16 fns (Module-Function-Count FAIL), 100-line apply_model (Function-LOC FAIL). It conflates the constructor cluster (start-up phase: build a ModelSpec from args) with the dispatcher cluster (per-item phase: apply a ModelSpec to data) -- the building-chains vs dispatching-chains anti-pattern from docs/loose-coupling.md.

Split into:

1. model_dispatch.rs (keep: constructor cluster, ~7 fns):
   - eval_linear, eval_embedding, eval_chain, eval_residual
   - eval_attention, eval_rms_norm, activation_kind
   (the eval_* model-DSL builders)

2. model_apply.rs (new file: dispatcher cluster, ~9 fns):
   - eval_apply, eval_predict_batch, eval_attention_weights
   - apply_model (the 100-LOC dispatcher; in this step further split apply_model's match arms into named helpers: apply_linear, apply_chain, apply_activation, apply_residual, apply_embedding, apply_linear_lora_arm. The orchestrator apply_model() becomes a thin dispatch <= 25 lines).
   - check_device_agreement
   - tokens_to_onehot + validate_token_id (move with apply_model since they're its inputs)

3. model_apply_attention.rs (new file: attention sub-cluster from the dispatcher side, ~6 fns):
   - apply_attn_head, apply_attention, apply_attention_rank2
   - slice_cols
   - extract_attn_weights + compute_attn_weights (already partially split in step 005 of saga 32)

Promote needed items to pub(crate) for cross-file access.

Target: model_dispatch.rs ~250 lines (PASS). model_apply.rs ~300 lines (PASS). model_apply_attention.rs ~350 lines (PASS). apply_model fn <= 25 lines (PASS). Retires 3 FAILs: file-LOC, fn-count, function-LOC on apply_model.

Strict gate: net-negative on BOTH fails AND warnings vs HEAD~1.