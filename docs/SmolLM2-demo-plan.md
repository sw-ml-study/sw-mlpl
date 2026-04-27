# SmolLM2 demo plan

## Goal

Build a LoRA-on-MLX fine-tuning demo around SmolLM2, starting with
the 135M variant and leaving 360M/1.7B as stretch targets. The demo
should show a realistic advantage over notebooks: reproducible
adapter training, small artifact output, remote MLX execution, and a
compiled inference application.

SmolLM2 is a compact language-model family with 135M, 360M, and 1.7B
parameter variants. The 135M model is the right initial target for a
local MLX demo.

## Why this is better than a notebook

- Training data, prompt template, LoRA config, optimizer settings, and
  evaluation prompts live in one replayable source file.
- The REPL can inspect adapters, loss curves, tokenized examples, and
  generated samples without hidden notebook state.
- Remote MLX lets the orchestrator stay on any host while adapter
  training runs on the Apple Silicon peer.
- The compiled app can ship only the inference path plus adapter load,
  turning the experiment into a small command users can run.
- Adapter artifacts are small and can be versioned separately from the
  base model.

## Repository strategy

Use the existing `softwarewrighter/efficient-llm` repo as the home
for demo-specific code, tokenizer/weight download instructions,
adapter artifacts, and example datasets. This plan is for adding an
sw-MLPL track to that repo, not for creating a new repo. Base weights,
tokenizer files, adapter artifacts, and example datasets should not
live in the MLPL language repo.

Suggested layout inside the existing repo:

- `mlpl/smollm2_lora.mlpl`
- `mlpl/smollm2_eval.mlpl`
- `data/train.jsonl`
- `data/eval.jsonl`
- `adapters/` ignored by default or managed with Git LFS
- `weights/` ignored, with download instructions
- `src/main.rs` for compiled inference wrapper

## MLPL support needed

1. Weight import.
   - Safetensors loader.
   - Tensor-name mapping for decoder-only transformer weights.
2. Tokenizer import.
   - Hugging Face tokenizer JSON or a minimal compatible BPE path.
3. Transformer inference.
   - Decoder block with causal attention, RMSNorm, RoPE if required,
     tied output projection, and KV-cache later.
4. LoRA over transformer projections.
   - Target modules: attention q/v first; then q/k/v/o and MLP gates.
   - Adapter save/load independent of base weights.
5. MLX execution.
   - Forward and adapter update path inside `device("mlx")`.
6. Compiled app.
   - First milestone can embed interpreter for inference.
   - Later milestone lowers enough transformer inference to Rust.

## Demo shape

### REPL/interpreter flow

```mlpl
tok = load_tokenizer("weights/smollm2-tokenizer.json")
base = load_safetensors_model("weights/SmolLM2-135M", "smollm2")
train_set = load_jsonl("data/train.jsonl")

adapter = lora(base, 8, 16.0, 0)

device("mlx") {
  train 200 {
    batch = sample_lm_batch(tok, train_set, 128, step)
    loss = causal_lm_loss(adapter, batch.input, batch.target)
    adam(loss, adapter, 0.0002, 0.9, 0.999, 0.00000001)
    loss
  }
}

save_lora(adapter, "adapters/demo.safetensors")
generate(adapter, tok, "Explain MLPL in one sentence:", 80)
loss_curve(last_losses)
```

### Compiled-app flow

```sh
mlpl build mlpl/smollm2_eval.mlpl -o target/smollm2-demo
target/smollm2-demo \
  --weights weights/SmolLM2-135M \
  --adapter adapters/demo.safetensors \
  --prompt "Explain MLPL in one sentence:"
```

## Phases

1. Import tokenizer and small safetensors fixtures.
2. Implement enough transformer inference for SmolLM2-135M.
3. Add LoRA target modules and adapter save/load.
4. Train a tiny adapter on MLX using remote peer.
5. Add eval prompts and before/after comparison.
6. Build compiled inference wrapper.
7. Keep demo-specific artifacts in `softwarewrighter/efficient-llm`
   and keep only reusable language/runtime support in `sw-mlpl`.

## Acceptance tests

- Tokenizer round-trips a small prompt fixture.
- Safetensors import maps known tensor names and shapes.
- Base model produces deterministic logits for a fixture.
- LoRA training changes only adapter parameters.
- Adapter load reproduces generated text for a fixed seed.
- The compiled app can run inference with a saved adapter.

## References

- SmolLM2 135M model card:
  <https://huggingface.co/HuggingFaceTB/SmolLM2-135M>
- SmolLM2 360M model card:
  <https://huggingface.co/HuggingFaceTB/SmolLM2-360M>
- SmolLM2 1.7B model card:
  <https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B>
- SmolLM2 paper: <https://arxiv.org/abs/2502.02737>
- Existing sw project repo:
  <https://github.com/softwarewrighter/efficient-llm>
