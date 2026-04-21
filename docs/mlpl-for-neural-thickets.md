# MLPL for Neural Thickets (design sketch)

> Status: design sketch. Not yet on a saga schedule. See
> `docs/plan.md` Saga 20 for the proposed work.

This doc explores what it would take to express
[Neural Thickets / nt-rs](https://github.com/swcraig/nt-rs)
(weight-perturbation specialists + top-K ensembling) as a single
MLPL program with a heatmap output. nt-rs is itself a Rust
research harness for the
[RandOpt](https://www.alphaxiv.org/overview/2603.12228.pdfv1)
approach: in pretrained models, the weight-space neighborhood is
dense with task-specialized solutions; sample N perturbations,
evaluate, take top-K, ensemble.

## What nt-rs does in one paragraph

Take a pretrained model. Generate N variants by adding structured
noise to its weights (Gaussian, low-rank, with optional layer
targeting -- attention-only, MLP-only, embed-and-head). Run each
variant on a task suite. Score. Pick the top-K specialists per
task category. Combine their outputs (best-of-k, majority vote,
confidence-weighted). Render heatmaps showing which perturbation
family specializes on which task.

## Where MLPL sits today

Saga 13 ships an end-to-end Tiny LM trained on a Shakespeare
snippet (`demos/tiny_lm.mlpl`). Saga 14 (in flight, v0.11.0
target) puts that same forward pass on Apple Silicon via MLX.
Together they give us a "base model" to perturb in MLPL source --
small enough that 20-50 variants are interactive on a laptop, big
enough to actually exhibit specialization patterns on the
character-level task.

The platform already owns most of the machinery a thicket needs:

- Forward + autograd: `apply(model, X)`, `cross_entropy`, `adam`,
  `train N { ... }` -- Sagas 9-13.
- Random noise: `randn(seed, shape)` -- Saga 8.
- Top-K logits: `top_k(logits, k)` -- Saga 13.
- Heatmap viz: `svg(matrix, "heatmap")` -- Saga 7.
- Per-axis scoring: `mean(losses, axis)` plus the labeled-axis
  surface from Saga 11.5.
- Device targeting: `device("mlx") { ... }` for the variant loop
  -- Saga 14.

What we do **not** have today:

1. **Model cloning.** Each call to `linear(in, out, seed)`
   allocates fresh params under a generated name. There is no
   builtin that says "give me an independent copy of `base` whose
   params live under new names." A user can simulate it by
   constructing a parallel model with a different seed offset, but
   that is "two random initializations," not "perturb one base."
2. **Per-family parameter walks.** Saga 11/13 names params with
   pattern-encoded handles (`__attn_Wq_3`, `__linear_W_2`,
   `__embed_E_0`), so the information needed to filter "all
   attention projections" is already there -- it is just not
   surfaced. nt-rs's `attention_only` / `mlp_only` / `embed_and_head`
   families are name-pattern walks waiting for a builtin.
3. **In-place add to a named param.** The autograd path mutates
   params via `adam` and `momentum_sgd`, but the Model DSL has no
   user-facing "add this delta to those params" operation outside
   an optimizer step.
4. **Index-returning top-K and per-row scatter.** `top_k(logits,
   k)` returns logits, not indices. We need either an
   `argtop_k(losses, k)` or a way to gather the K best variant
   indices for the ensemble step.

These gaps are small (each is roughly one builtin, ~50-150 LOC) but
real. They are the work item for Saga 20.

## Strawman MLPL source

The program below is the target "after Saga 20 ships." Lines that
require a not-yet-shipped builtin are tagged with `# NEW`. The
language surface is otherwise current MLPL.

```mlpl
# demos/neural_thicket.mlpl -- Saga 20 demo.
#
# Train a base Tiny LM, generate N perturbation variants across
# four families, score each on held-out tokens, take top-K,
# ensemble, render a [family x seed] specialization heatmap.

# ---- 1. Base model + tokenizer (Saga 13 surface) ----
corpus = load_preloaded("tiny_shakespeare_snippet")
tok    = train_bpe(corpus, 280, 0)
ids    = apply_tokenizer(tok, corpus)
X      = reshape(shift_pairs_x(ids, 32),
                 [reduce_mul(shape(shift_pairs_x(ids, 32)))])
Y      = reshape(shift_pairs_y(ids, 32),
                 [reduce_mul(shape(shift_pairs_y(ids, 32)))])

V = 280 ; d = 32 ; h = 1
base = chain(
  embed(V, d, 0),
  residual(chain(rms_norm(d), causal_attention(d, h, 1))),
  residual(chain(rms_norm(d), linear(d, 128, 2),
                              relu_layer(),
                              linear(128, d, 3))),
  rms_norm(d),
  linear(d, V, 4))

experiment "thicket_base" {
  train 200 {
    adam(cross_entropy(apply(base, X), Y),
         base, 0.001, 0.9, 0.999, 0.00000001);
    loss_metric = cross_entropy(apply(base, X), Y)
  }
}

# ---- 2. Perturb on MLX, four families x four seeds ----
device("mlx") {
  to_device(base, "mlx")

  # Held-out validation tokens for scoring.
  val_ids = apply_tokenizer(tok, "to be or not to be that is the question")
  val_X   = shift_pairs_x(val_ids, 16)
  val_Y   = shift_pairs_y(val_ids, 16)

  N        = 16
  sigma    = 0.02
  families = ["mlp_only", "attention_only", "embed_and_head", "all_layers"]
  losses   = zeros([N])

  repeat N {
    fam     = families[step / 4]                       # NEW: string-array indexing
    variant = clone_model(base)                        # NEW: deep copy w/ fresh names
    perturb_params(variant, fam, sigma, step + 100)    # NEW: family-targeted randn add
    losses  = scatter(losses, step,                    # NEW: scalar scatter at index
                      cross_entropy(apply(variant, val_X), val_Y))
  }

  # ---- 3. Top-K specialists ----
  k        = 4
  best_idx = argtop_k(neg(losses), k)                  # NEW: index-returning top-K

  # ---- 4. Ensemble: average logits from the K best variants ----
  ens_logits = zeros(shape(apply(base, val_X)))
  for i in best_idx {                                  # iterate index list
    v = clone_model(base)
    perturb_params(v, families[i / 4], sigma, i + 100)
    ens_logits = ens_logits + apply(v, val_X)
  }
  ens_logits = ens_logits / k
  ens_metric = cross_entropy(ens_logits, val_Y)
}

# ---- 5. Heatmap: rows = family, cols = seed ----
heat : [family, seed] = reshape(losses, [4, 4])
svg(heat, "heatmap")
```

The non-`NEW` parts of this program are valid MLPL today. The four
`NEW` lines collapse to the proposed builtins listed below.

## Proposed builtins (Saga 20 surface)

1. **`clone_model(m) -> Model`.** Deep-copies a model: walks its
   `ModelSpec` tree, allocates fresh-named params with the same
   shapes and values, returns a new `ModelSpec` with the new
   names. Independent of the original; mutating one does not
   affect the other.

2. **`perturb_params(m, family, sigma, seed)`.** Walks `m`'s
   params, filters by family (the same name patterns currently
   used internally: `__attn_*` for attention, `__linear_*` for
   MLP / output head, `__embed_*` for embeddings), and adds
   `sigma * randn(seed, shape)` to each in place.

3. **`argtop_k(values, k) -> Vec<index>`.** Index-returning
   companion to the existing `top_k(logits, k)` (which returns
   masked logits suitable for sampling). Used to drive the
   ensemble loop.

4. **`scatter(buffer, index, value)`.** In-place scalar write at
   `index` into a rank-1 array. Pairs naturally with `repeat N`
   loops where each iteration produces one number.

Layer-targeted families that match nt-rs's surface:

- `all_layers` -- every param.
- `attention_only` -- `__attn_Wq_*`, `__attn_Wk_*`, `__attn_Wv_*`,
  `__attn_Wo_*`.
- `mlp_only` -- `__linear_W_*` and `__linear_b_*` outside the
  final projection head.
- `embed_and_head` -- `__embed_E_*` plus the final projection
  layer's `W` / `b`.
- `early_N_layers` / `late_N_layers` -- requires either explicit
  layer index in the param name (which we do not encode today)
  or a depth-aware walk of the `ModelSpec` tree. Defer to a
  follow-up unless trivial.

Two-line variants of `gaussian_weight` and `low_rank_delta` (from
nt-rs's perturbation taxonomy) are both implementable on top of
`perturb_params`: the Gaussian case is the default; low-rank can
land as an optional `rank=` kwarg or as a sibling builtin
`perturb_low_rank(m, family, rank, seed)` once the family walker
exists.

## Why this fits the platform thesis

MLPL's promise is "a few lines of array notation express a
non-trivial ML idea, with first-class visualization." A 60-line
nt-rs port that produces a specialization heatmap is exactly the
kind of artifact that argues the thesis the way the Saga 13 Tiny
LM did. It also exercises the MLX backend at a workload it is
genuinely good at (many independent forward passes -- lazy graph
fusion across variants), without needing distributed execution
(which is Saga 17's CUDA story, not this).

## Non-goals

- **No real pretrained LLM.** The demo runs on the Saga 13 Tiny
  LM. nt-rs at production scale wants Llama-class models with
  Ollama or a Python sidecar; that requires either a checkpoint
  format (Saga 15+) or a sidecar protocol (Saga 19's REST tool
  integration).
- **No distributed coordination.** nt-rs supports a coordinator
  + worker shard pattern. That belongs to Saga 17 (CUDA + LAN
  training). Saga 20 stays single-process.
- **No new optimizer.** This is an evaluation-only loop; `adam`
  trains the base, then variants are scored on a frozen forward.
- **No new viz primitive.** Existing `svg(..., "heatmap")` and
  `loss_curve` cover the visualization story. Specialization
  rankings render as a sorted bar chart via `svg(..., "bar")`.

## Where this lands on the saga timeline

After Saga 14 ships (`v0.11.0`, MLX backend) the variant loop is
fast enough to be interactive. Saga 15 (LoRA + quantization)
overlaps mechanically with `perturb_params` -- both want a clean
"walk model params, transform them" abstraction -- but the LoRA
work is on the train-time path while perturbation is on the
inference-time path, so they can ship in either order.

The natural slot is after Saga 19 (LLM REST tool integration), as
**Saga 20 -- Perturbation research demos**, with `neural_thicket`
as the headline demo and the four builtins above as the language
surface delta. See `docs/plan.md` for the entry.
