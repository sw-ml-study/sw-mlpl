# BDH demo plan

## Goal

Use the Baby Dragon Hatchlings repo to build an sw-MLPL track for
Dragon Hatchling (BDH), the architecture described in "The Dragon
Hatchling: The Missing Link between the Transformer and Models of the
Brain." BDH is presented as an attention-based state-space sequence
learning architecture with a scale-free, biologically inspired graph
structure, sparse positive activations, and an interpretability story
around state and synaptic plasticity.

The demo should start with a tiny BDH-inspired sequence model rather
than a full reproduction. The strongest version is a compact language
or grid-sequence model:

- base "hatchling" sequence model trained from scratch on tiny text or
  symbolic grid data;
- LoRA adapters for domain or behavior specialization;
- MLX acceleration for adapter training;
- REPL visualizations for graph activity, sparse state, and adapter
  effects.

## Why this is better than a notebook

- One source file can define the world, policy, reward, adapter, and
  visualization.
- Users can train a new hatchling personality live in the REPL without
  re-running notebook cells in the right order.
- LoRA adapters are small enough to save, compare, and swap.
- The compiled app can become a tiny interactive toy: load a base
  model, choose an adapter, and watch the agent act.
- The interpretability angle is first-class: sparse activations,
  graph neighborhoods, and concept-specific state changes can be
  rendered as normal MLPL outputs instead of ad-hoc notebook plots.

## Repository strategy

Use the existing `softwarewrighter/bdh` repo as the home for paper
notes, architecture experiments, saved adapters, interactive app code,
and artifacts. This plan is for adding an sw-MLPL track to that repo,
not for creating a new repo. Treat `pathwaycom/bdh` as the upstream
reference implementation and paper companion.

Suggested layout inside the existing repo:

- `mlpl/bdh_tiny_base.mlpl`
- `mlpl/bdh_lora_domain.mlpl`
- `mlpl/bdh_state_probe.mlpl`
- `assets/` for graph/state visualizations
- `src/` for compiled app wrapper
- `adapters/` ignored or Git LFS-managed if published

## MLPL support needed

1. Adapter persistence.
   - `save_lora(model, "path")`
   - `load_lora(model, "path")`
2. Tiny RL or imitation helpers.
   - `sequence_dataset(name, n, context)`
   - `causal_lm_loss(model, X, Y)`
   - Optional: grid-sequence tasks later.
3. Better demo asset handling.
   - SVG graph visualization and simple raster export.
4. Model comparison helpers.
   - `compare_generation(base, adapted, prompts)`
   - sparse activation heatmaps.
   - graph neighborhood overlays.
5. Compiled app wrapper.
   - Interactive mode can embed the interpreter first.
6. BDH-inspired primitives.
   - Sparse positive activation helpers.
   - Graph-neighborhood mixing or attention/state-space hybrid layer.
   - State probe output for interpretability demos.

## Demo shape

### REPL/interpreter flow

```mlpl
tok = tokenizer("byte")
data = sequence_dataset("tiny_stories_slice", 256, 64)
brain = bdh_tiny(vocab_size(tok), hidden=64, graph_k=8, seed=0)

device("mlx") {
  train 100 {
    batch = sample_lm_batch(tok, data, 64, step)
    logits = apply(brain, batch.input)
    loss = causal_lm_loss(logits, batch.target)
    adam(loss, brain, 0.003, 0.9, 0.999, 0.00000001)
    loss
  }
}

domain = lora(brain, 4, 8.0, 1)
device("mlx") {
  train 40 {
    batch = sample_lm_batch(tok, data, 64, step)
    logits = apply(domain, batch.input)
    loss = causal_lm_loss(logits, batch.target)
    adam(loss, domain, 0.01, 0.9, 0.999, 0.00000001)
    loss
  }
}

svg(bdh_graph(brain), "graph")
svg(bdh_state_probe(domain, "dragon"), "heatmap")
```

### Compiled-app flow

```sh
mlpl build mlpl/bdh_demo.mlpl -o target/bdh-demo
target/bdh-demo --adapter adapters/domain.safetensors --prompt "Once"
```

## Phases

1. Paper-to-MLPL notes: identify the minimum BDH-inspired layer that
   can be expressed with current MLPL plus small new primitives.
2. CPU tiny sequence model on byte-level or synthetic text.
3. MLX training for the base model.
4. LoRA adapters for domain specialization.
5. Adapter save/load.
6. Graph and state-probe visualizations.
7. Compiled interactive app.

## Acceptance tests

- A tiny BDH-inspired model improves next-token loss over a small MLP
  or vanilla recurrent baseline.
- A LoRA adapter changes generation or task behavior without mutating
  the base model.
- Sparse state and graph visualizations are deterministic for a fixed
  seed.
- The same demo has REPL and compiled-app commands.

## References

- The Dragon Hatchling: The Missing Link between the Transformer and
  Models of the Brain: <https://arxiv.org/abs/2509.26507>
- Upstream BDH implementation: <https://github.com/pathwaycom/bdh>
- Existing sw project repo:
  <https://github.com/softwarewrighter/bdh/commits/main/>
