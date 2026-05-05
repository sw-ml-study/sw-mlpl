# MLPL Course Outline: From Zero to Training and Distilling LLMs

A learning path for absolute beginners that uses the MLPL platform's
shipped demos, glossary entries, and tutorial lessons to walk through
the concepts and history of machine learning, all the way up to
training a transformer language model from scratch, fine-tuning it
with LoRA, distilling it to a smaller student, and using it as a
tool. Along the way the outline calls out where MLPL already has the
material, where it has glossary coverage but no runnable demo, and
where MLPL is missing the concept entirely.

## How to read this outline

Each module lists, in order:

- **Concepts** -- the ideas and history a student should walk away
  with, anchored to entries in `docs/glossary.md`.
- **Tutorial lessons** -- existing inline lessons in
  `apps/mlpl-web/src/lessons.rs` and `lessons_advanced.rs` that match
  the module.
- **Demos** -- the runnable `.mlpl` programs in `demos/` (and the
  matching scripted walkthroughs in `docs/demos-scripts.md`) that
  exercise the concepts.
- **Gaps** -- concepts, glossary entries, tutorials, or demos that
  are missing from MLPL today and would have to be authored to make
  the module fully self-contained.

A "[deferred]" tag marks items the platform consciously punted to a
future saga. An "[out of scope]" tag marks items unlikely to land in
MLPL's teaching language at all (for example, full RLHF). A "[new]"
tag marks items that have never been planned and are pure suggestions.

The reading-time estimates assume a complete novice working through
the web REPL at <https://sw-ml-study.github.io/sw-mlpl/>.

## Audience and prerequisites

- **Target reader:** comfortable with high-school algebra, has used a
  programming language before, has never trained a neural network.
- **Tooling:** any modern browser for the web REPL; optionally a
  terminal with the MLPL REPL installed (Apple Silicon for the MLX
  modules).
- **Time budget:** roughly 25-35 hours end to end if every demo is
  run and every "try it" prompt is attempted.

---

## Module 0 -- Orientation and platform tour (1 hour)

### Concepts

- What MLPL is: a Rust-first array language for ML inspired by APL,
  J, and BQN; one source runs in the interpreter, on MLX, or compiled
  to native / WASM.
- Why an array language for teaching ML: shapes are first-class,
  every concept has a one-line spelling.
- The REPL as a workspace -- inspired by APL's `)WSID`, `)VARS`.

### Tutorial lessons

- "Hello Numbers" (lesson 1)
- "Workspace Introspection" -- introduces `:vars`, `:wsid`,
  `:describe`, `:builtins`.

### Demos

- `demos/basics.mlpl` -- the platform's "do everything" smoke test.
- "Demo 3 -- Web REPL tour" in `docs/demos-scripts.md`.

### Gaps

- **[new]** A "What is ML, and why are we here?" lesson that names
  the destination (training and shaping language models) so the
  early algebra-heavy lessons feel motivated. Today the tutorial
  jumps from arithmetic to arrays without telling the student where
  the road ends.
- **[new]** A short "history of ML" lesson: Perceptron (1958) ->
  Backprop (1986) -> CNNs / AlexNet (2012) -> RNN/LSTM era ->
  Attention (2017) -> GPT / ChatGPT era. The glossary already has
  `Perceptron`, `RNN / LSTM / GRU`, `Transformer`, `GPT`, `BERT`,
  `Emergent Behavior`, but no narrative ties them together.

---

## Module 1 -- Arrays, shapes, and the language core (2-3 hours)

### Concepts (glossary)

- `Tensor`, `Rank (of a tensor)`, `shape / rank (builtins)`, `Axis`,
  `Dense (array, layer)`, `iota (builtin)`, `reshape (builtin)`,
  `Transpose`, `fill / zeros / ones (builtins)`, `randn / random
  (builtins)`, `BuiltinRef (:foo syntax)`, `reduce (builtin)`,
  `reduce_add / reduce_mul (builtins)`.

### Tutorial lessons

- "Arrays"
- "Variables"
- "Built-in Functions"
- "Matrices"
- "Named Axes"

### Demos

- `demos/matrix_ops.mlpl`
- `demos/computation.mlpl`

### Gaps

- **[scheduled]** A dedicated "Reductions and broadcasting" lesson.
  `docs/missing-demos.md` already flags this -- broadcasting rules
  are used everywhere, never taught explicitly.
- **[scheduled]** A "Shape manipulation" lesson covering `reshape`,
  `transpose`, `shape`, `rank`, `iota` together instead of as side
  notes inside ML lessons.
- **[new]** Glossary entry for *broadcasting* -- the concept appears
  in many demos but has no glossary line.

---

## Module 2 -- Linear algebra and math primitives (2 hours)

### Concepts (glossary)

- `Matmul`, `Dot product`, `pow (builtin)`, `sqrt (builtin)`, `log
  (builtin)`, `abs (builtin)`, `mean (builtin)`, `Sigmoid`, `Tanh`,
  `ReLU`, `Softmax`, `Activation function`,
  `Comparison ops: gt, lt, eq (builtins)`, `concat (builtin)`,
  `last_row (builtin)`, `repeat block (language keyword)`,
  `for / in (language keyword)`.

### Tutorial lessons

- "Linear Algebra"
- "Math and Activations"
- "Comparisons and Logic"
- "Loops and Iteration"

### Demos

- `demos/repeat_demo.mlpl`

### Gaps

- **[new]** A "broadcasting cheat sheet" worked example. Repeats the
  Module 1 gap from a numerical-ops angle.
- **[new]** Glossary entry for *cosine similarity* -- referenced as
  "normalized dot product" inside `Dot product` but no entry of its
  own. Comes up in retrieval / RAG.

---

## Module 3 -- Data, tokenization, and visualization (2 hours)

### Concepts (glossary)

- `load / load_preloaded (builtins)`, `Token`, `Tokenizer`,
  `BPE (Byte-Pair Encoding)`, `tokenize_bytes / decode_bytes
  (builtins)`, `Vocabulary`, `Sequence`, `shuffle (builtin)`,
  `batch / batch_mask (builtins)`, `Padding`, `Heatmap`,
  `hist (builtin)`, `scatter_labeled (builtin)`,
  `boundary_2d (builtin)`, `grid (builtin)`,
  `Train / Validation / Test Split`, `Test set`, `Validation set`.

### Tutorial lessons

- "Loading Data"
- "Tokenizing Text"
- "Visualizing Data"
- "Visualizing Analyses"

### Demos

- `demos/loss_curve.mlpl`
- `demos/decision_boundary.mlpl`
- `demos/analysis_demo.mlpl`

### Gaps

- **[scheduled]** "Synthetic datasets" lesson grouping `randn`,
  `blobs`, `moons`, `circles` together (called out in
  `docs/missing-demos.md`).
- **[scheduled]** "Decision boundaries" as a first-class lesson
  rather than a name-drop inside "Visualizing Analyses".
- **[new]** A "Data hygiene" lesson covering `Data Leakage`,
  `Distribution Shift`, `Shortcut Learning / Spurious Correlations`
  -- glossary entries exist; no lesson stitches them together.
- **[deferred]** `Data Augmentation` glossary entry exists but is
  not a builtin; would need a small "augment by perturbation"
  worked example.

---

## Module 4 -- Classical ML: linear, logistic, k-means, PCA (3-4 hours)

### Concepts (glossary)

- `Classifier`, `Logits`, `Cross entropy`, `Labels`,
  `One-hot encoding`, `MSE (Mean Squared Error)`, `Confusion matrix`,
  `Decision boundary`, `Clustering`, `K-Means`,
  `pairwise_sqdist (builtin)`, `PCA (Principal Component Analysis)`,
  `power iteration`, `Bias-Variance Tradeoff`,
  `Overfitting / Underfitting`, `Optimization vs Generalization`,
  `Manifold Hypothesis`, `Curse of Dimensionality`.

### Tutorial lessons

- "Machine Learning: Logistic Regression"
- "Unsupervised: K-Means"
- "Dimensionality Reduction: PCA"
- "Multi-class Classification"

### Demos

- `demos/logistic_regression.mlpl`
- `demos/kmeans.mlpl`
- `demos/pca.mlpl`
- `demos/softmax_classifier.mlpl`

### Gaps

- **[deferred]** `Cross-Validation` glossary entry exists but no
  builtin; tutorial could write k-fold by hand once `split` /
  `val_split` are taught.
- **[deferred]** `ROC / AUC`, `Precision vs Recall`, `Calibration`
  glossary entries exist but no demos exercise them.
- **[new]** Glossary entries / lesson coverage for *F1* and
  *threshold tuning* -- lightly implied by precision/recall but not
  named.

---

## Module 5 -- Neural networks: MLPs, autograd, optimizers (4 hours)

### Concepts (glossary)

- `Perceptron`, `MLP (Multi-Layer Perceptron)`,
  `Universal Approximation`, `Forward pass`, `Backward pass`,
  `Backpropagation`, `Chain rule`, `Autograd`, `Gradient`,
  `Gradient descent`, `param / tensor (constructors)`, `Parameter`,
  `Frozen (parameter)`, `Optimizer`, `Adam`, `Momentum SGD`,
  `Learning rate`, `Learning Rate Schedules`, `Warmup`, `Step`,
  `Epoch`, `Hyperparameter`, `Train (loop)`,
  `Weight Initialization`.

### Tutorial lessons

- "Going Non-Linear: A Tiny MLP"
- "Automatic Differentiation"
- "Optimizers and Schedules"

### Demos

- `demos/tiny_mlp.mlpl`
- `demos/moons_mlp.mlpl`
- `demos/circles_mlp.mlpl`

### Gaps

- **[new]** A "Why backprop?" historical lesson that names Werbos
  (1974), Rumelhart-Hinton-Williams (1986), and the AI winter that
  preceded them. Glossary covers the algorithm; the history is
  absent.
- **[deferred]** `Dropout`, `Batch Normalization`, `Layer norm`,
  `Gradient Clipping`, `Weight Decay`, `Label Smoothing`,
  `Early Stopping` -- all glossary entries exist with `[deferred]`
  notes; none have demos. A "regularization tour" lesson could be
  authored once two or three of those builtins ship.
- **[deferred]** `Stop gradient / detach` -- glossary acknowledges
  the gap; no builtin yet.

---

## Module 6 -- Composing models, experiments, typing (2 hours)

### Concepts (glossary)

- `Model`, `Layer`, `Linear (layer)`, `Chain (Model DSL)`,
  `Residual`, `RMS Norm`, `tanh_layer / relu_layer / softmax_layer`,
  `experiment block`, `:experiments`, `:vars`, `:models`, `:tags`,
  `:wsid`, `:describe`, `Workspace`, `Loss`, `Probability`.

### Tutorial lessons

- "Model Composition (the Model DSL)"
- "Experiments"
- "Workspace Introspection"
- "Typed ML Values" (advanced lessons set)

### Demos

- `demos/feasibility.mlpl`
- `demos/trace_demo.mlpl`

### Gaps

- **[deferred]** `Checkpointing` -- glossary entry calls out that
  MLPL has no checkpoint format; a "save / resume training" demo is
  the obvious follow-up.
- **[new]** A lesson on `estimate_train` /
  `estimate_hypothetical` / `feasible` / `calibrate_device`
  (Saga 22 builtins). Glossary covers them in one combined entry but
  there is no tutorial that walks a student through "is this model
  feasible on my laptop?". The `feasibility.mlpl` demo exists; a
  tutorial does not.

---

## Module 7 -- Attention and transformer mechanics (4-5 hours)

### Concepts (glossary)

- `Attention`, `Self-attention`, `Causal attention`, `Cross-attention`,
  `Multi-head attention`, `Head (attention)`,
  `Scaled dot-product attention`, `Query (Q)`, `Key (K)`, `Value (V)`,
  `Projection (matrix)`, `Attention map`, `Mask`,
  `Positional encoding`, `Embedding`, `embed_table (builtin)`,
  `Feedforward (FFN)`, `Transformer`, `Transformer block`,
  `Decoder / encoder`, `Encoder-decoder`.

### Tutorial lessons (web tutorial advanced set)

- "Attention Patterns"
- "Self-Attention from Scratch"
- "Multi-Head Attention from Scratch"
- "Cross-Attention from Scratch"
- "Encoder Block"
- "Decoder Block"

### Demos

- `demos/attention.mlpl`
- `demos/transformer_block.mlpl`

### Gaps

- **[deferred]** `RoPE (Rotary Positional Embeddings)`,
  `KV Cache`, `Flash Attention`, `GQA (Grouped Query Attention)` --
  all glossary entries exist; no tutorials and no demos. Each is a
  natural standalone lesson once the underlying primitive ships.
- **[deferred]** `Layer norm` proper (MLPL ships only `RMS Norm`); a
  comparison tutorial would be illustrative.
- **[deferred]** `Convolution` / `CNN` -- relevant in the historical
  arc that leads to attention; deferred along with `conv2d`.
- **[deferred]** `RNN / LSTM / GRU` -- glossary acknowledges these
  are deferred; without a recurrent layer there is no RNN demo, so
  the historical "transformers replaced RNNs" story is told but not
  shown.

---

## Module 8 -- Training a tiny LM from scratch (4-5 hours)

### Concepts (glossary)

- `LM (Language Model)`, `GPT`, `Pretraining / fine-tuning`,
  `Context Window`, `Sampling`, `Top-k`, `Temperature`,
  `Sequence`, `Beam search`, `shift_pairs_x / shift_pairs_y
  (builtins)`, `last_row (builtin)`, `concat (builtin)`,
  `Perplexity`, `Scaling Laws`.

### Tutorial lessons

- "Language Model Basics"
- "Training and Generating"

### Demos

- `demos/tiny_lm.mlpl`
- `demos/tiny_lm_generate.mlpl`
- `docs/demos-scripts.md` Demo 1 ("Platform thesis in 10 lines").

### Gaps

- **[planned]** A SmolLM2-scale pretrain or fine-tune demo; the plan
  exists in `docs/SmolLM2-demo-plan.md` but the demo is not yet
  shipped. Until it is, the student sees only character-scale
  training.
- **[deferred]** `Beam search`, `Speculative Decoding`, `KV Cache` as
  inference tutorials. Glossary entries are present; tutorials and
  builtins are not.
- **[deferred]** `Mixed Precision` and *gradient accumulation* --
  required to train at non-toy scale; the former has a glossary
  entry, the latter has none.
- **[new]** A standalone "Evaluation: perplexity, held-out loss,
  qualitative samples" lesson. `Perplexity` is in the glossary;
  there is no lesson that runs an eval pass.
- **[new]** Glossary entries for *BLEU*, *ROUGE*, *MMLU*, *HellaSwag*
  and similar benchmarks. None today.

---

## Module 9 -- Embeddings, representations, and retrieval (2 hours)

### Concepts (glossary)

- `Embedding`, `embed_table (builtin)`, `Latent Space`,
  `Representation Learning`, `t-SNE`, `PCA (Principal Component
  Analysis)`, `knn (builtin)`, `pairwise_sqdist (builtin)`,
  `RAG (Retrieval-Augmented Generation)`,
  `Interpretability / Mechanistic Interpretability`.

### Tutorial lessons

- "Embedding exploration" (advanced lessons set)

### Demos

- `demos/embedding_viz.mlpl`
- `docs/demos-scripts.md` Demo 9 ("Embedding visualization").

### Gaps

- **[deferred]** A full RAG pipeline demo. Glossary entry exists;
  Saga 16 left a real RAG flow as future work. A tutorial that
  embeds a small corpus, indexes it with `knn`, and threads the
  result into `llm_call` would close the loop.
- **[deferred]** UMAP -- glossary mentions it as a "deferred follow-
  up" inside `Manifold Hypothesis` and t-SNE notes.
- **[new]** A circuits / mechanistic-interpretability tutorial.
  `Superposition` and `Mechanistic Interpretability` entries exist;
  no lesson exercises them on a trained tiny LM.

---

## Module 10 -- Fine-tuning, freezing, and LoRA (3 hours)

### Concepts (glossary)

- `Fine-tuning`, `Frozen (parameter)`, `unfreeze (builtin)`,
  `LoRA (Low-Rank Adaptation)`, `Transfer Learning`,
  `Catastrophic Forgetting`, `Pretraining / fine-tuning`.

### Tutorial lessons

- "LoRA Fine-Tuning" (advanced lessons set)

### Demos

- `demos/lora_finetune.mlpl`
- `demos/lora_finetune_mlx.mlpl`
- `docs/demos-scripts.md` Demo 8 ("LoRA fine-tune").

### Gaps

- **[deferred]** `Quantization` -- glossary entry exists; a QLoRA
  tutorial is the obvious next step once a quantization layer ships.
- **[deferred]** A full SFT (supervised fine-tune on instruction
  data) tutorial: prompt formatting, masking the prompt out of the
  loss, evaluation with held-out instructions. Glossary covers
  `Few-shot Learning` and `Prompting` but not SFT specifics.
- **[deferred]** `Weight Decay` / `AdamW` -- common in fine-tuning,
  glossary deferred.
- **[new]** Glossary entries for *instruction tuning*, *chat
  template*, *prompt masking*, *gradient accumulation*. None today.

---

## Module 11 -- Distillation (3 hours, almost entirely a gap)

### Concepts (glossary)

- `Distillation`
- adjacent: `Temperature`, `Logits`, `Cross entropy`,
  `LogProbability`, `Soft targets` (not yet a glossary entry).

### Tutorial lessons

- None today.

### Demos

- None today.

### Gaps (this is mostly missing surface)

- **[deferred]** A distillation tutorial covering teacher logits,
  soft targets at high temperature, KL-divergence loss, and
  comparison to plain cross-entropy on hard labels. Saga 18 in
  `docs/missing-demos.md` plans MLPL distillation pipelines; v0.19
  defers them.
- **[new]** A `kl_div(p_logits, q_logits, temperature)` builtin and
  matching glossary entry. Today MLPL has `cross_entropy` but no
  KL-divergence helper, so a distillation loss has to be built by
  hand from `softmax`, `log`, and `reduce_add`.
- **[new]** A `soft_targets(logits, temperature)` helper or
  glossary entry. Currently the concept lives only in research
  papers, not in MLPL.
- **[new]** A "DistilBERT-style" or "tiny-LM teacher / smaller-LM
  student" demo: train the Saga 13 LM as a teacher, train a smaller
  student against the teacher's softened logits, compare student
  perplexity vs from-scratch baseline at the same size.
- **[new]** A "self-distillation" worked example (student is the
  same architecture as the teacher) since it is one of the simplest
  ways to introduce the concept without a second model.
- **[new]** Glossary entries for *online distillation*, *offline
  distillation*, and *born-again networks*.

---

## Module 12 -- Alignment, preference learning, safety (mostly out of scope today)

### Concepts (glossary)

- `RLHF (Reinforcement Learning from Human Feedback)`,
  `DPO (Direct Preference Optimization)`, `Constitutional AI`,
  `Preference Learning`, `Reward Hacking`, `Goodhart's Law`,
  `Hallucination`, `Jailbreaks`, `Prompt Injection`,
  `OOD Inputs (Out-of-Distribution)`, `Calibration`,
  `Uncertainty Estimation`.

### Tutorial lessons

- None today.

### Demos

- None today.

### Gaps

- **[out of scope today]** RLHF / DPO pipelines. Glossary entries
  flag these as out of scope for the current MLPL surface (no RL,
  no preference data structures). A teaching outline can still
  *describe* them and point students at the published recipes.
- **[new]** A "preference-data primitives" sketch: paired
  completions, ranking loss (`logsigmoid` of preferred minus
  rejected reward). The glossary entry for `Preference Learning`
  defines the math; nothing in MLPL implements it.
- **[new]** A safety / jailbreak demo using `llm_call` -- show how
  prompt injection slips past a naive system prompt. Glossary has
  `Prompt Injection` and `Jailbreaks` but no exercise.
- **[new]** A calibration / uncertainty-quantification tutorial that
  composes `confusion_matrix` with the `Neural Thicket` ensemble to
  produce confidence intervals.

---

## Module 13 -- Robustness, ensembling, interpretability, research (3 hours)

### Concepts (glossary)

- `Ensembling`, `clone_model (builtin)`,
  `perturb_params (builtin)`, `scatter (builtin)`,
  `Adversarial examples`, `Loss Landscape`,
  `Lottery Ticket Hypothesis`, `Grokking`, `Double Descent`,
  `Emergent Behavior`, `Inductive Bias`,
  `Mechanistic Interpretability`, `Superposition`.

### Tutorial lessons

- "Neural Thickets" (advanced lessons set)

### Demos

- `demos/neural_thicket.mlpl`
- `demos/neural_thicket_mlx.mlpl`
- `docs/demos-scripts.md` Demo 7 ("Neural Thickets").

### Gaps

- **[planned]** TRM and HRM demos (`docs/TRM-demo-plan.md`,
  `docs/HRM-demo-plan.md`) bring recursive / hierarchical reasoning
  to the curriculum. Plans exist; demos are not shipped.
- **[planned]** BDH demo (`docs/BDH-demo-plan.md`) introduces
  state-space sequence learning, sparse positive activations, and
  biological-graph priors. Plan exists; demo is not shipped.
- **[planned]** TTT (`docs/TTT-demo-plan.md`) introduces test-time
  training; same status.
- **[deferred]** Adversarial-input demos (current MLPL only does
  weight-space perturbation, per the glossary entry on
  `Adversarial examples`).
- **[new]** A circuits / activation-patching tutorial -- the
  natural follow-up to `Mechanistic Interpretability` once the
  embedding tutorial is digested.

---

## Module 14 -- Tools, inference, deployment (3 hours)

### Concepts (glossary)

- `llm_call (builtin)`, `In-Context Learning (ICL)`,
  `Few-shot Learning`, `Chain of Thought`, `Tool Use`,
  `Prompting`, `Hallucination`, `device block (language keyword)`,
  `Inference`.

### Tutorial lessons

- "Running on MLX"

### Demos

- `demos/llm_tool.mlpl`
- `demos/mlx_remote.mlpl`
- `demos/tiny_lm_mlx.mlpl`
- `docs/demos-scripts.md` Demo 2 (MLX), Demo 6 (compile-to-Rust).

### Gaps

- **[deferred]** A full tool-use loop (LLM produces JSON, MLPL
  executes a function, result feeds back). `Tool Use` glossary
  entry calls this out; only single-shot `llm_call` ships today.
- **[deferred]** A browser-side `llm_call` story; v0.19 is CLI-only
  due to CORS / proxy.
- **[new]** A "deploying a trained model" tutorial that ties
  together Saga 22 feasibility estimation, `cargo run -p mlpl-build`
  to native, and WASM cross-compile. The pieces are documented in
  `docs/compiling-mlpl.md` but no lesson stitches them into a
  capstone.
- **[new]** Glossary entries for *agent loop*, *function calling*,
  *system prompt*, *context stuffing*. Today these only show up
  obliquely under `Prompting` and `Tool Use`.

---

## Module 15 -- Capstone project: train, fine-tune, distill, deploy (5 hours)

A multi-session project that uses the modules above end to end. The
student should be able to:

1. **Train** a tiny transformer LM from scratch on a short corpus
   (Module 8, `tiny_lm.mlpl` as the template).
2. **Evaluate** it with held-out perplexity and a sampled completion
   (Module 8 with the missing eval lesson filled in).
3. **Fine-tune** it with LoRA on a small instruction-shaped corpus
   (Module 10, `lora_finetune.mlpl` as the template).
4. **Distill** the LoRA-fine-tuned teacher into a smaller student
   (Module 11 -- *requires the missing distillation surface*).
5. **Inspect** the result with `embed_table`, `tsne`, and `knn`
   (Module 9).
6. **Deploy** by compiling the student to a native binary and a WASM
   bundle (Module 14 plus `docs/compiling-mlpl.md`).

### Gaps that block the capstone today

- Step 4 cannot run end to end until the distillation surface from
  Module 11 ships. The teaching language can stop at "here is how
  you would express it" but cannot demonstrate it.
- Step 2 lacks a packaged eval lesson; the student writes the
  perplexity computation by hand from `cross_entropy` plus `exp`.
- A single saved-checkpoint format would let steps 1, 3, and 6 hand
  artifacts off cleanly. `Checkpointing` is glossary-deferred.

---

## Cross-cutting gaps summary

These appear under multiple modules above; pulling them together
into a single list makes the priority easier to see.

### Tutorial lessons that are missing or would meaningfully unblock the path

1. "Reductions and broadcasting" (already in `docs/missing-demos.md`).
2. "Shape manipulation" (already in `docs/missing-demos.md`).
3. "Synthetic datasets" (already in `docs/missing-demos.md`).
4. "Decision boundaries" (already in `docs/missing-demos.md`).
5. "What is ML, and why are we here?" -- destination-setting intro.
6. "A short history of ML" -- Perceptron through Transformers.
7. "Why backprop?" -- historical complement to "Automatic
   Differentiation".
8. "Feasibility and budget" -- tutorial wrapper around
   `estimate_train` / `feasible` / `calibrate_device`.
9. "Evaluating a language model" -- perplexity, held-out loss,
   qualitative samples.
10. "Distillation: teacher logits and soft targets" -- new module
    centerpiece.
11. "Saving and resuming training" -- the missing capstone glue.
12. "Deploying a trained model" -- compile-to-native and WASM.
13. "Building a RAG pipeline" -- the deferred Saga 16 follow-up.
14. "Activation patching and circuits" -- mechanistic
    interpretability lesson.

### Demos that are missing

- Distillation demo (teacher / student / KL-divergence loss).
- A KV-cache / efficient-decoding demo.
- A QLoRA / quantization demo once a quantization layer ships.
- A full SFT-on-instructions demo (today only LoRA exists).
- A RAG-end-to-end demo.
- A safety / prompt-injection demo against `llm_call`.
- An adversarial-input demo (input space, not weight space).
- Recurrent-network demo (RNN / LSTM / GRU) for the historical arc.
- CNN demo for the same reason.
- Calibration / uncertainty-quantification demo built on top of
  `Neural Thickets`.
- The planned but unshipped reasoning demos: TRM, HRM, BDH, TTT,
  SmolLM2 fine-tune.

### Glossary entries that should be added

- Broadcasting (the rules, not just by example).
- Cosine similarity (currently folded into `Dot product`).
- F1 score / threshold tuning.
- BLEU, ROUGE, MMLU, HellaSwag (or a single "LM benchmarks" entry).
- KL divergence.
- Soft targets.
- Online vs offline distillation.
- Born-again networks (self-distillation).
- Instruction tuning, chat template, prompt masking.
- Gradient accumulation.
- Agent loop, function calling, system prompt.
- AdamW (separate from `Adam` to make weight decay explicit).

### Builtins or runtime features that would close the biggest gaps

These are not part of the course outline itself, but the outline's
gaps trace back to them. Most are already labelled `deferred` in the
glossary.

- `kl_div(p_logits, q_logits, temperature)` and a matching
  `soft_targets`. **Module 11 blocker.**
- A checkpoint save/restore format. **Module 6 / capstone blocker.**
- `stop_gradient(x)` / `detach(x)`. **Module 5.**
- `dropout`, `layer_norm`, `weight_decay` / `adamw`,
  `gradient_clip`. **Module 5 regularization tour.**
- `quantize(model, bits)` and a matching QLoRA path. **Module 10.**
- `kv_cache(model)` and a `generate_with_cache` helper.
  **Module 8 / 14.**
- `rope_encoding(T, d)`. **Module 7.**
- A first-class `Distribution` type (Saga 24 plan). Unblocks VAE,
  preference-learning probability ops, and Bayesian uncertainty
  bands.

---

## Suggested per-week pacing

A four-to-six-week pace for a self-directed learner:

- **Week 1.** Modules 0-2 (the language and math primitives).
- **Week 2.** Modules 3-4 (data, classical ML, evaluation basics).
- **Week 3.** Modules 5-6 (neural networks, autograd, model DSL).
- **Week 4.** Modules 7-8 (attention, transformers, training a tiny
  LM end to end).
- **Week 5.** Modules 9-10 (embeddings, retrieval, LoRA fine-tune).
- **Week 6.** Modules 11-15 (distillation, alignment overview,
  research demos, deployment, capstone). Many gaps land here; treat
  the week as a survey if those gaps are still open.

---

## Related docs

- `docs/glossary.md` -- normative concept dictionary; every concept
  in this outline has an entry there.
- `apps/mlpl-web/src/lessons.rs` and `lessons_advanced.rs` -- the
  shipped tutorial.
- `demos/` -- the shipped runnable demos.
- `docs/demos-scripts.md` -- scripted walkthroughs for the most
  presentable demos.
- `docs/missing-demos.md` -- the original audit that this outline
  builds on; several of the "scheduled" gaps above come from there.
- `docs/SmolLM2-demo-plan.md`, `docs/HRM-demo-plan.md`,
  `docs/TRM-demo-plan.md`, `docs/BDH-demo-plan.md`,
  `docs/TTT-demo-plan.md` -- planned-but-unshipped advanced demos
  that this outline references in Modules 8 and 13.
- `docs/plan.md` -- saga roadmap; the deferred items above mostly
  trace back to entries in its "Future saga sequence".
