# Glossary

Short definitions for the ML terms that appear in MLPL demos
and lessons. Alphabetical. Each entry names the closest MLPL
construct so you can poke at the concept in the REPL. Concepts
that MLPL does not ship today carry a `deferred` note pointing
at the saga that may add them; concepts outside MLPL's
teaching-language scope (MLOps, deployment, monitoring) carry
an `out of scope` note.

## abs (builtin)

Elementwise absolute value: `abs(x)` returns `|x|` for each
element. Pure scalar map; preserves shape.

## Activation function

A non-linear elementwise function applied to a layer's output
so a stack of layers can represent non-linear behavior. MLPL
ships `tanh_layer()`, `relu_layer()`, `softmax_layer()`,
plus the math primitives `sigmoid`, `tanh_fn`.

## Adam

An adaptive gradient-descent optimizer that tracks per-
parameter first and second moments (`m`, `v`) and rescales
each update by their ratio. Spelled `adam(loss, params, lr,
b1, b2, eps)` in MLPL.

## AdamW

Adam with decoupled weight decay. Plain `adam` mixes
weight decay into the gradient before the moment update,
which interacts oddly with the per-parameter learning-rate
scaling; AdamW applies the decay directly to the parameter
update step, keeping it independent from the moment
scaling. Empirically robust default for transformer
training. **Deferred** in MLPL: `adam(loss, params, lr,
b1, b2, eps)` ships today; an `adamw` variant with explicit
`weight_decay` is on the regularization-tour roadmap.

## Adversarial examples

Inputs crafted with small, often imperceptible perturbations
that flip a model's prediction. The classic image-recognition
example: change a few pixels and "panda" becomes "gibbon".
MLPL ships `perturb_params(model, family, sigma, seed)` for
weight-space perturbation (Saga 20); input-space adversarial
attacks are deferred.

## Agent loop

A control loop where an LLM repeatedly emits a tool-use
request, the runtime executes the tool, the result is fed
back into the prompt, and the cycle continues until a stop
condition. The "function calling" interface is what the
model sees; the loop is what the application implements.
MLPL: `llm_call(url, prompt, model)` is one step of such
a loop; building the loop itself is application code.

## Attention

A layer that computes a weighted average of one set of values
where the weights come from comparing a query against a set
of keys. The classic formula is `softmax(Q @ K^T / sqrt(d)) @
V`. MLPL builds it into `attention(d_model, heads, seed)`
and `causal_attention(...)`; the manual three-line version
runs in the "Attention Pattern" demo.

`apply(mdl, X)` and `attention_weights(mdl, X)` accept both
rank-2 `[seq, d_model]` input and rank-3 `[B, T, d_model]`
batched input for single-head models (Saga 29 step 008).
For rank-3 input each batch entry is processed independently
and the per-batch outputs are stacked back. Multi-head
(`heads > 1`) plus rank-3 still rejects -- that combination
lands in Saga 29 step 010.

## Attention map

The `[T, T]` matrix of attention weights between every pair
of positions in a sequence. Renders cleanly as a heatmap.
Returned by `attention_weights(model, tokens)` in MLPL.

## Autoencoder

A network trained to reconstruct its input through a low-
dimensional bottleneck (latent) layer. Used for compression,
denoising, and unsupervised representation learning. See also
`VAE`. Not a v0.19 MLPL builtin; deferred.

## Autograd

Reverse-mode automatic differentiation. The runtime records
a tape during the forward pass; calling `grad(loss, wrt)`
walks the tape backwards to compute the gradient with respect
to a tracked parameter.

## Axis

A dimension of a tensor. A `[batch, vocab]` tensor has
axis 0 (batch) and axis 1 (vocab). Many ops take an `axis`
argument: `softmax(x, 1)` normalizes along axis 1; named
axes (`x : [batch, vocab] = ...`) make this self-documenting.

## Backpropagation

The algorithm for computing gradients of a scalar loss with
respect to every parameter, by walking the computation graph
backwards from the loss and applying the chain rule at each
node. Foundation of every gradient-based ML technique. In
MLPL the algorithm is hidden behind `grad(loss, wrt)`: the
forward pass records a tape, `grad` walks it backward and
returns the gradient with the same shape as `wrt`. The
"Automatic Differentiation" tutorial lesson covers the
mechanics; the "Logistic Regression" and "Tiny MLP" lessons
show the manual chain-rule version (`dZ = pred - y`,
`dW = X^T dZ / N`) that `grad` automates. See also Backward
pass, Autograd, Chain rule.

## Backward pass

The traversal of the autograd tape that computes gradients.
Runs implicitly when you call `grad`, `adam`, or
`momentum_sgd`. Synonymous with backpropagation.

## Batch

A group of inputs processed together so a single forward pass
amortizes the per-step overhead. Datasets reshape into
`[batch, ...]` arrays; `batch(x, size)` slices a dataset.
Batch SIZE is a hyperparameter -- larger smooths gradients
but costs more memory.

## Batch Normalization

Normalize each feature across the batch axis so its mean is
zero and variance is one, then apply a learnable scale and
shift. Stabilizes training of deep networks; sensitive to
small batch sizes. MLPL ships `rms_norm` (a simpler scheme);
batch norm is deferred.

## batch / batch_mask (builtins)

`batch(x, size)` slices a `[N, ...]` array into a `[B, size,
...]` array of batches; the trailing batch is zero-padded if
`size` does not divide `N`. `batch_mask(x, size)` returns the
matching `[B, size]` `0 / 1` mask so downstream ops can ignore
the padded positions.

## Beam search

A decoding strategy that keeps the top-`k` partial sequences
at each step instead of committing to one. MLPL's generation
demos use greedy / multinomial sampling via `sample` + `top_k`
rather than beam search; beam search is a deferred follow-up.

## BERT

A specific encoder-only transformer architecture trained with
masked-token prediction. Pre-2020-era foundation for many
classification/QA tasks. MLPL builds tiny LMs in the demo
suite but does not ship pretrained model weights; loading
external checkpoints is a future saga.

## Bias

The constant additive term in `y = x @ W + b`. A trainable
1-D parameter, smaller than the weight matrix; auto-tagged
`Bias` in v0.19.

## Bias-Variance Tradeoff

Total expected error decomposes into: bias (model class is
too simple to fit the truth), variance (model is too sensitive
to the training set), and irreducible noise. Increasing model
capacity drops bias but raises variance; regularization,
ensembling, and more data shift the optimum.

## Born-again networks (self-distillation)

A simplified distillation setup where the student has the
*same architecture* as the teacher. Train teacher to
convergence; then train a fresh-init student against the
teacher's softened logits with KL-divergence loss; often
the student reaches slightly better accuracy than the
teacher despite identical capacity. The simplest demo of
distillation because there is no architecture mismatch to
explain. **Deferred** in MLPL: blocked on `kl_div` /
`soft_targets` builtins.

## BPE (Byte-Pair Encoding)

A subword tokenizer that starts from raw bytes and greedily
merges the most-frequent adjacent pair until a target vocab
size is reached. MLPL ships `train_bpe(corpus, vocab_size,
seed)` plus `apply_tokenizer` and `decode`.

## blobs / circles / moons (builtins)

Synthetic 2-D classification datasets. `blobs(seed, n,
centers)` returns `[N, 3]` rows of `[x, y, label]` for `n`
points around each given center. `circles(seed, n, noise)`
makes concentric noisy rings; `moons(seed, n, noise)` makes
the classic two-half-moon pattern. All three appear in the
"Decision Boundary" / "K-Means" / "Moons MLP" demos.

## boundary_2d (builtin)

Renders a 2-D classifier's decision surface as an SVG
heatmap with the training points overlaid. Signature:
`boundary_2d(predictions, grid_shape, points, labels)`.

## Broadcasting

The rules for combining arrays of different shapes in
elementwise operations: a singleton axis (size 1) stretches
to match the other operand's size on that axis; missing
trailing axes are treated as singletons. `[3, 4] + [4]`
broadcasts the rank-1 vector across the rows of the rank-2
matrix; `[3, 4] + [3, 1]` broadcasts the column-vector
across the columns. MLPL applies broadcasting to all
binary arithmetic ops (`+`, `-`, `*`, `/`) and the
comparison builtins (`gt`, `lt`, `eq`). Mismatches surface
as `EvalError::ShapeMismatch` whose Display shows both
labeled shapes side by side. See `docs/lang-reference.md`
"Broadcasting Rules" for the full table.

## BuiltinRef (`:foo` syntax)

A first-class-ish reference to a builtin or operator. Written
as `:` immediately followed by an identifier or one of
`+ * / -`. Examples: `:add`, `:max`, `:+`, `:sigmoid`. Used
as the first arg to higher-order builtins like `reduce(:op,
x[, axis])`. Lives in a separate namespace from regular
variables, so `add = 42` does not shadow `:add`. Forward-
compatible with first-class functions: when `Value::Function`
lands, `:foo` lifts to a function value.

## Calibration

How well a model's reported confidence matches its actual
accuracy. A well-calibrated classifier that says "70% sure"
is right 70% of the time. Modern neural nets are typically
overconfident. Temperature scaling on logits is the standard
post-hoc fix. Deferred in MLPL.

## Catastrophic Forgetting

When fine-tuning a model on new data erases what it learned
from prior data. Mitigations: rehearsal (mix old + new
batches), elastic weight consolidation, low-rank adapters
(LoRA preserves the base by freezing it). MLPL's
`freeze` + `lora` workflow is the simplest defense.

## Causal attention

Self-attention with a lower-triangular mask before softmax so
position `t` cannot peek at `t+1`. Required for autoregressive
language models. MLPL: `causal_attention(d_model, heads,
seed)`.

## Chain (Model DSL)

Sequential composition of layers. `chain(linear(2, 8, 0),
tanh_layer(), linear(8, 2, 1))` is a 2-layer MLP.

## Chain rule

The calculus identity `d(f(g(x)))/dx = f'(g(x)) * g'(x)`,
applied recursively to compose gradients across the layers
of a neural network. Backpropagation IS the chain rule
applied to the computation graph: at each node, multiply the
incoming gradient by the local Jacobian, then propagate
upstream. MLPL's `grad(loss, wrt)` does this automatically;
the "Tiny MLP" lesson shows the manual two-layer version
where the hidden-layer gradient is `dZ1 = (dZ2 W2^T) * (1 -
H * H)` -- the tanh derivative being the local Jacobian
factor. See also Backpropagation, Autograd.

## Chain of Thought

A prompting technique that asks an LLM to produce intermediate
reasoning steps before its final answer, often improving
accuracy on multi-step problems. MLPL ships `llm_call(url,
prompt, model)` for general LLM tool-use; CoT is a prompting
pattern, not a builtin.

## Chat template

The string-formatting convention that wraps multi-turn
chat messages with role markers (`<|user|>`, `<|assistant|>`,
`<|system|>`, etc.) before tokenization, so the model can
distinguish who said what. Different model families ship
different templates (ChatML, Llama, Alpaca); using the
wrong one degrades responses sharply. **Deferred** in
MLPL: `apply_tokenizer(tok, text)` is byte-level only;
template handling is application code.

## Checkpointing

Saving model state (parameters + optimizer state) periodically
so a long training run can resume if interrupted. Distinct
from "gradient checkpointing" which trades compute for memory
during backprop. MLPL doesn't ship a checkpoint format yet;
`experiment "name" { body }` records run metadata only.

## Classifier

A model whose output is a discrete class. The architecture
ends in a `[batch, num_classes]` logit matrix; training uses
`cross_entropy(logits, y)`. See "Linear Softmax Classifier"
demo.

## CLIP (Contrastive Language-Image Pre-training)

A dual-encoder model trained on (image, caption) pairs. One
encoder embeds images, another embeds text; both project
into a shared vector space where matched pairs have high
cosine similarity. The contrastive loss pushes matched
pairs together and unmatched pairs apart. Enables
zero-shot image classification: compare a query image to a
list of text prompts. Foundation for many vision-language
systems. **Deferred** in MLPL: needs image inputs + the
dual-encoder training pattern.

## Clustering

Grouping points without labels. K-means is the classic
algorithm; the demo runs ten Lloyd-iteration steps over 90
points and three centroids.

## clone_model (builtin)

`clone_model(m)` deep-copies a `ModelSpec` tree, allocating a
fresh disjoint set of param names. Saga 20's "Neural Thicket"
ensembling clones a base model into 16 disjoint variants
before perturbing each.

## CNN (Convolutional Neural Network)

A network built around convolution and pooling layers,
designed for grid-structured data (images). MLPL does not
ship a `conv2d` layer today -- transformer + MLP families are
the v0.19 model surface; convolutional layers are a deferred
follow-up.

## Confusion matrix

A `[K, K]` table where row `i`, column `j` counts how often a
class-`i` ground truth was predicted as class `j`. Diagonals
are correct predictions. MLPL: `confusion_matrix(preds, y)`
returns the matrix; `svg(_, "heatmap")` renders it.

## Constitutional AI

Training an LLM via human-written principles ("be helpful,
honest, harmless") plus self-critique cycles. A specific RLHF
variant. Out of MLPL's current scope.

## Context Window

The maximum sequence length a model can attend over in a
single forward pass. Limited by the `[T, T]` attention
matrix's quadratic memory cost. The Tiny LM demos use
`block=8`. Larger context demands KV cache, sparse attention,
or state-space alternatives.

## Convolution

The window-and-sum operation at the core of CNNs. Not a
v0.19 builtin in MLPL; deferred along with CNN-family demos.

## Cosine similarity

A similarity measure between two vectors that is invariant
to magnitude: `cos(u, v) = dot(u, v) / (norm(u) * norm(v))`,
range `[-1, 1]`. Distinct from `Dot product` (which is
sensitive to magnitude). The standard scoring function for
embedding-based retrieval (RAG) and nearest-neighbor lookup
on learned representations. MLPL: build from `dot`,
`sqrt`, `reduce_add` -- no dedicated builtin.

## Comparison ops: `gt`, `lt`, `eq` (builtins)

Elementwise predicates returning `0.0` / `1.0`. `gt(a, b)`,
`lt(a, b)`, `eq(a, b)`. MLPL has no boolean type -- the
`0 / 1` floats double as masks (multiply to filter) and
counts (`reduce_add` to sum a "how many true" tally).

## concat (builtin)

`concat(a, b)` stitches two rank-1 vectors end to end. Used
in the Tiny LM generation loop to grow the prompt
sequence by appending each newly-sampled token.

## Cross-attention

Attention where the queries come from one sequence and the
keys / values come from a different sequence. Same math as
self-attention, just with Q from a different source than K
and V. The weight matrix is `[T_query, T_source]` -- non-
square, distinguishing it from self-attention's `[T, T]`.
Demos: "Cross-Attention from Scratch" (full pipeline) and
"Attention Pattern" (the two-Q/K-matrix variant).

## Cross entropy

The classification loss used with logits. Numerically stable
implementation in MLPL: `cross_entropy(logits, y)` where
logits are `[N, V]` and y are `[N]` integer class indices.
Auto-tags the result as `Loss(CrossEntropy)`.

## Cross-Validation

Splitting the training set into `k` folds and training `k`
models each with one fold held out, to estimate generalization
without a separate validation set. Standard in classical ML;
expensive for deep nets, so usually replaced by a single
train/val/test split. MLPL ships `split` / `val_split` for
the simpler pattern.

## Curriculum Learning

Ordering training data from easy to hard so the model masters
simple patterns before tackling complex ones. MLPL's `for row
in ds { body }` lets you control iteration order; curriculum
is a discipline applied on top, not a builtin.

## Curse of Dimensionality

In high dimensions, points become roughly equidistant, volume
concentrates near the surface of any hypercube, and density
estimation becomes intractable. Drives the value of structured
priors, dimensionality reduction (PCA, t-SNE), and learned
representations.

## :describe (REPL command)

`:describe <name>` prints a typed summary of a binding -- shape
+ tag + values preview for an array, layer tree for a model,
vocab + merge count for a tokenizer, signature for a builtin.
Saga 23 v0.19 added per-tag bodies (Probability rows show
the verified-or-violated row-sum invariant; Gradient shows
`wrt`, etc.).

## device block (language keyword)

`device("target") { body }` pushes a device target onto a
stack so ops inside the body dispatch through that backend.
MLPL: `device("cpu") { ... }` (default), `device("mlx") {
... }` (Apple MLX, Saga 14), and -- with `--peer` registered
-- a remote service peer (Saga R1). Bindings created inside
the block carry the device tag forward; cross-device ops
strict-fault.

## Data Augmentation

Generating extra training samples by transforming existing
ones (flipping images, masking tokens, paraphrasing text)
without changing the label. Cheap regularization. Not a
v0.19 MLPL builtin; deferred.

## Data Leakage

When test-set information sneaks into training -- via
duplicates, time-ordered splits violated, target-derived
features, or evaluation on data seen during preprocessing.
The most common silent killer of "great" benchmark numbers.
Defense: cross-check the split; never look at test until
the final report.

## Data Parallel Training

The simplest distributed-training scheme. Replicate the
model on N devices; each device processes one Nth of the
batch in parallel; gradients are all-reduced across
devices before applying the optimizer step. Scales batch
size linearly with device count. Distinct from tensor
parallel (split a single layer's weights across devices)
and pipeline parallel (split layers across devices).
**Deferred** in MLPL: the `device("mlx") { ... }` peer
dispatch is the building block, but no automatic
gradient all-reduce.

## Decision boundary

The surface in input space that separates one predicted class
from another. `boundary_2d(predictions, grid_shape, points,
labels)` renders it for a 2-D classifier.

## Decision Tree

A tree of yes/no splits on input features that classifies
or regresses by traversing from root to leaf. Each
internal node tests one feature against a threshold; the
leaf carries the prediction. Trained greedily by picking
the split that maximizes information gain (or minimizes
Gini impurity). Easy to interpret, prone to overfitting --
the standard fix is ensembling, e.g. Random Forest.
**Deferred** in MLPL: needs a tree data structure and a
greedy split fitter.

## Decoder / encoder

In a transformer, the encoder builds a sequence of
representations from the input; the decoder generates output
tokens conditioned on encoder outputs and prior generations.
MLPL's "Tiny LM" demos build a decoder-only stack
(causal attention + MLP head).

## Dense (array, layer)

Stored as a contiguous row-major buffer with no zero-skipping
(no sparse representation). MLPL's `DenseArray` is the only
array shape today.

## Diffusion Models

Generative models that learn to denoise a sequence of
increasingly noisy versions of the data. State of the art for
images / video. Out of MLPL's current scope; loading a
trained checkpoint is the entry point most users want, which
is a separate saga.

## Distillation

Training a smaller "student" model to imitate a larger
"teacher" model's outputs (logits or probabilities) instead
of the original labels. Compresses knowledge into faster /
cheaper models. Saga 18 plans MLPL distillation pipelines;
deferred in v0.19.

## Distribution Shift

When test-time inputs differ statistically from training
inputs. Includes covariate shift (input distribution changes,
label given input is stable) and concept drift (label
function itself changes). Models trained without explicit
robustness assume no shift; performance silently degrades.

## Distillation -- see entry above

## Dot product

The sum of elementwise products of two equal-length vectors:
`dot(a, b) = sum(a * b)`. The fundamental building block of
linear algebra: matmul is just dot products at scale, cosine
similarity is a normalized dot product, and attention scoring
is a scaled dot product. MLPL: `dot(a, b)` for vectors;
`matmul(A, B)` for matrices.

## Double Descent

Empirical phenomenon where test error first drops, rises
(near the interpolation threshold), then drops again as model
capacity grows past it. Conflicts with the classical
bias-variance picture; central to modern overparameterized
deep learning theory.

## DPO (Direct Preference Optimization)

A simpler alternative to RLHF that fits a model directly to
human preferences without training a separate reward model
or running reinforcement learning. Out of MLPL's current
scope; preference-data builtins are a future saga.

## Dropout

A regularization technique that zeros out a random fraction
of activations during training so the network learns
distributed representations. Not a v0.19 MLPL builtin;
deferred.

## Early Stopping

Halting training when validation loss stops improving (rather
than after a fixed step count) to avoid overfitting. MLPL
doesn't ship a built-in early-stop hook; the user can
condition the `train` body on `last_losses` patterns.

## embed_table (builtin)

`embed_table(model)` walks a `ModelSpec` tree depth-first
left-to-right and returns the first `Embedding` layer's
`[vocab, d_model]` matrix as a plain array. Saga 16.5
shipped this so demos can inspect / project / cluster a
learned embedding after training.

## Embedding

A learned `[vocab, d_model]` lookup table that maps token ids
to dense vectors. MLPL: `embed(vocab, d_model, seed)` is a
Model DSL layer; `embed_table(model)` returns the underlying
`[vocab, d_model]` matrix.

## Emergent Behavior

A capability that appears in a large model but is absent or
broken in a smaller one of the same architecture, sometimes
appearing abruptly with scale (e.g. arithmetic, reasoning).
Heavily debated whether truly emergent or an artifact of
metric thresholding. MLPL's tiny scales don't produce
emergent capabilities; the lessons are pedagogical.

## Encoder-decoder

A two-stack transformer where the encoder turns an input
sequence into contextualized representations and the decoder
generates an output sequence by attending both to its own
prior tokens (causal self-attention) and to the encoder's
output (cross-attention). Used for sequence-to-sequence
tasks like translation. Decoder-only models (GPT-style) skip
the encoder; encoder-only models (BERT-style) skip the
decoder. See "Decoder / encoder" for the role of each side.

## Ensembling

Running multiple trained models on the same input and
combining their outputs (averaging logits, voting). Often
beats any single member at the cost of inference time and
memory. MLPL's "Neural Thicket" demo (Saga 20) runs a 16-
member weight-perturbed ensemble end-to-end.

## Epoch

One full pass through the training dataset. Distinct from
"step" -- a step is one optimizer update. Epochs are dataset-
relative; steps are gradient-relative.

## estimate_train / estimate_hypothetical / feasible / calibrate_device (builtins)

Saga 22's "feasibility" surface. `estimate_train(model, steps,
batch_size, seq_len[, dtype_bytes])` returns a `[5]` array
`[params, vram_bytes, disk_bytes, flops, wall_seconds]` from
a `ModelSpec`. `estimate_hypothetical(name, ...)` answers the
same question for SmolLM / Llama / Qwen scale points without
materializing weights. `feasible(estimate, [vram, disk,
wall])` returns a `0 / 1` guard for `if feasible(est,
budget) { train ... }`. `calibrate_device()` runs a 1024x1024
matmul benchmark and caches device GFLOPS for honest
estimates.

## :experiments (REPL command)

`:experiments` lists every recorded experiment block in the
session merged with on-disk records (terminal REPL only).
Each row shows the experiment name + captured `_metric`-
suffixed scalars; pairs with `compare(a, b)` for delta
inspection.

## experiment block (language keyword)

`experiment "name" { body }` runs the body and captures any
scalar variable whose name ends in `_metric` as a metric on
an `ExperimentRecord`. The record lands in
`env.experiment_log` (always) and on disk under
`<exp_dir>/<name>/<ts>/run.json` (terminal REPL only).
Use to pin a reproducible notebook entry per run.

## F1 score / threshold tuning

The harmonic mean of precision and recall: `F1 = 2 * P * R
/ (P + R)`. Single number that penalizes either failure
mode. For a probabilistic classifier, the threshold that
maximizes F1 is rarely 0.5 -- you sweep thresholds against
held-out predictions and pick the argmax. MLPL: build from
`gt(probs, threshold)` masks plus `confusion_matrix` or
`reduce_add` over true-positive / false-positive counts; no
dedicated builtin.

## Feedforward (FFN)

The two-linear-layer subblock inside a transformer block:
`linear(d_model, d_ff) -> relu_layer() -> linear(d_ff,
d_model)`. Hidden width `d_ff` is typically `4 * d_model`.
Provides the position-wise nonlinear transformation between
attention layers. In MLPL: `chain(linear(d, d_ff, s1),
relu_layer(), linear(d_ff, d, s2))`.

## Few-shot Learning

Doing a task by showing the model a handful of input/output
examples in its context (no gradient updates). The
"In-Context Learning" sibling concept. Pure prompting in MLPL
via `llm_call(url, prompt, model)`; no dedicated builtin.

## Fine-tuning

Continuing training from a pretrained model on a new dataset
or task. Often paired with `freeze` so only a subset of
parameters move. MLPL: `lora(model, rank, alpha, seed)` is
the parameter-efficient form (Saga 15).

## Flash Attention

A re-implementation of attention that fuses the softmax with
the matmuls and tiles to keep working memory in fast SRAM,
reducing both wall-clock and memory cost without changing
the math. MLPL's MLX backend uses naive attention today;
fused / flash variants are deferred.

## fill / zeros / ones (builtins)

Constant-array constructors. `zeros([d0, d1, ...])` makes a
zero-filled tensor of the given shape; `ones(...)` is one-
filled; `fill(shape, value)` is the general form. Used to
allocate accumulators (`losses = zeros([16])`), bias inits,
mask scaffolding.

## for / in (language keyword)

`for row in dataset { body }` streams over rows (or batches)
of a dataset, binding the row to `row` per iteration. The
last value of `body` is captured into `last_rows` for
plotting / inspection. Saga 12 added this construct; the
"Loading Data" tutorial walks it end-to-end.

## Function calling

A model-output convention where, instead of a free-text
answer, the LLM emits a structured tool-call (function name
+ JSON arguments). The application parses, executes, and
feeds the result back. Strictly an output-formatting
contract; the model is doing next-token prediction
underneath. Paired with the agent loop. MLPL: not a
builtin; achievable today by formatting prompts to request
JSON and parsing the response from `llm_call`.

## Forward pass

The traversal from inputs to outputs (no gradients computed).
Runs whenever you `apply(model, X)` or evaluate a math
expression.

## Frozen (parameter)

A parameter the optimizer skips during the update step. The
gradient still flows through during backprop; only the
weight update is suppressed. MLPL: `freeze(model)` /
`unfreeze(model)` from Saga 15.

## Goodhart's Law

"When a measure becomes a target, it ceases to be a good
measure." A core RLHF / alignment hazard: optimizing toward
any imperfect reward will eventually game it.

## GPT

A specific decoder-only transformer family popularized by
OpenAI. The "Tiny LM" demos build a 1-layer GPT-style stack
(embed + causal attention + RMS norm + linear head).

## GQA (Grouped Query Attention)

Compromise between multi-head attention (every head has its
own K, V) and multi-query attention (all heads share one K,
V). Reduces KV cache memory at little quality cost. MLPL's
`attention(d_model, heads, seed)` is full multi-head;
GQA / MQA are deferred.

## Gradient

The partial derivative of a loss with respect to a parameter.
Computed by `grad(loss, wrt_param)`. Auto-tags as
`Gradient(wrt=W1)` so `:describe g` shows which param it
belongs to.

## Gradient accumulation

Sum gradients over several mini-batches before applying an
optimizer step, simulating a larger effective batch size
than fits in memory at once. `effective_bs = micro_bs *
accumulation_steps`. Used routinely at LLM training scale
where the desired batch size exceeds device memory.
**Deferred** in MLPL: `train` block applies the optimizer
every iteration; an explicit accumulator pattern is
buildable but not first-class.

## Gradient Clipping

Capping the L2 norm of the gradient vector before applying
the optimizer update. Prevents explosive updates from rare
huge-gradient batches. Not a v0.19 MLPL builtin; deferred.

## Gradient descent

Updating parameters in the direction opposite the gradient.
Vanilla form: `W = W - lr * gW`. The "AND Logistic Regression"
and "Tiny MLP" demos write this loop out by hand;
production demos use `adam` or `momentum_sgd`.

## Grokking

Empirical phenomenon where training accuracy plateaus near
100% for many steps, then validation accuracy suddenly jumps
from chance to near-perfect. Suggests a phase transition from
memorizing to generalizing. Most clearly seen in toy modular
arithmetic.

## Hallucination

When an LLM produces fluent text that is factually wrong, with
no signal to the user that it's unreliable. Sometimes
mitigated by RAG (ground in retrieved facts) or explicit
"I don't know" training. Out of MLPL's current scope;
relevant when using `llm_call`.

## grid (builtin)

`grid([x_lo, x_hi, y_lo, y_hi], n)` returns an `[n*n, 2]`
array of `(x, y)` points evenly spaced over the rectangle.
Used by `boundary_2d` to query a classifier's surface for
decision-boundary plots.

## Head (attention)

One of the parallel attention components in multi-head
attention. Each head has its own Q/K/V projections of width
`d_k = d_model / heads` and operates independently; the
per-head outputs are concatenated to recover the full
`d_model` width. Different heads can specialize on different
relationships in the input. MLPL: `attention(d_model, heads,
seed)` with `heads >= 1`. See "Multi-Head Attention from
Scratch" for a per-head walkthrough.

## Heatmap

`svg(matrix, "heatmap")` renders a `[N, M]` array as a 2-D
intensity grid. Standard for attention maps and confusion
matrices.

## hist (builtin)

`hist(values, bins)` renders a histogram of a flat values
vector with the requested number of bins as an SVG. One-line
distribution inspection.

## Hyperparameter

A configuration value that is NOT learned by gradient descent
but chosen by the user: learning rate, batch size, number of
epochs, model dimensions (`d_model`, `heads`, `vocab_size`),
optimizer betas, etc. MLPL doesn't have a hyperparameter
sweep DSL today; `experiment "name" { body }` records what
was tried.

## Instruction tuning

Supervised fine-tuning on (instruction, response) pairs --
typically formatted with a chat template -- to teach a
pretrained base model to follow user instructions instead
of completing free text. The "SFT" step in the standard
pretrain -> SFT -> RLHF pipeline. **Deferred** in MLPL:
LoRA fine-tuning ships, but instruction-tuning datasets
require chat-template handling and prompt masking that
are not first-class.

## In-Context Learning (ICL)

The ability of a large model to learn a task purely from
examples shown in the prompt, with no gradient updates. The
core LLM superpower. Not a builtin in MLPL; use `llm_call`
to compose few-shot prompts.

## Inductive Bias

The set of assumptions a model class encodes about which
hypotheses are more likely. Convolutions assume spatial
locality + translation equivariance; transformers assume
all-to-all attention; linear models assume a linear
relationship. Stronger biases learn faster from less data
but cap the function class.

## Inference

Running a trained model on new inputs without updating
weights. In MLPL, just call `apply(model, X)` outside of
`train { ... }` -- no gradient tape, no optimizer step.

## Interpretability / Mechanistic Interpretability

Mechanistic interpretability is the program of reverse-
engineering trained neural networks circuit by circuit:
identifying attention heads, feature directions, and
algorithms the model implements. The "Embedding exploration"
demo is a tiny taste -- t-SNE / k-NN over a learned
embedding table. Full circuit-level work is out of MLPL's
v0.19 scope.

## Jailbreaks

Prompt patterns that trick an LLM out of its safety training
("ignore previous instructions", role-play attacks). LLM-
safety territory; MLPL exposes `llm_call` but doesn't ship
a jailbreak / safety-eval surface.

## iota (builtin)

`iota(n)` returns the integer sequence `[0, 1, ..., n-1]` as
a rank-1 vector. The most basic array constructor; building
block for indexing / shape arithmetic / one-hot scaffolding.

## Key (K)

One of the three projections in attention. Each token emits
a key advertising "what I have to offer"; the dot product `Q
@ K^T` measures how strongly each query matches each key,
producing the unnormalized score matrix. In MLPL:
`K = matmul(X, Wk)` where `Wk` is `[d_model, d_model]` for
single-head or `[d_model, d_k]` per head. Paired with Query
(Q) and Value (V).

## K-Means

Unsupervised clustering by alternating "assign each point to
its nearest centroid" and "move each centroid to the mean of
its assigned points". The K-Means demo runs ten iterations.

## knn (builtin)

`knn(X, k)` returns each row's `k` nearest non-self neighbors
sorted by ascending distance with lower-index tie-break.
`[N, k]` integer-index output. Saga 16 ships this for
embedding inspection.

## KV Cache

The cache of past Key and Value tensors a transformer keeps
during autoregressive decoding so each new token only needs
to compute its own row, not the entire `[T, T]` attention
matrix. MLPL's "Tiny LM Generate" demo recomputes from
scratch each step; KV cache is a deferred efficiency win.

## KL divergence

A non-symmetric measure of how one probability distribution
diverges from another: `KL(P || Q) = sum(P * (log(P) -
log(Q)))`, zero iff `P == Q`, larger when Q gives low
probability to events P deems likely. The natural
distillation loss: a high-temperature student softmax
trained to match a teacher's softened logits via KL.
**Deferred** in MLPL: build by hand from `softmax`,
`log`, `reduce_add`; a `kl_div(p_logits, q_logits,
temperature)` builtin is on the Module 11 distillation
roadmap.

## Label Smoothing

Replacing one-hot targets with a softer distribution (e.g.
`0.9` for the true class, `0.1 / (K-1)` for the rest) during
cross-entropy training. Reduces overconfidence and slightly
regularizes. Not a v0.19 MLPL builtin; deferred.

## Labels

Ground-truth integer class indices for a classification task.
A `Labels { num_classes }` tag in v0.19 carries the class
count so `confusion_matrix` and `cross_entropy` can validate
shape compatibility.

## last_row (builtin)

`last_row(m)` returns the last row of a rank-2 matrix as a
rank-1 vector. The Tiny LM generation loop uses it to pick
the next-token logits from `apply(model, seq)`'s `[T, V]`
output.

## Latent Space

The internal representation space a model builds, typically
a hidden layer's `[batch, d_model]` activations. Often more
semantically structured than the raw input -- nearest
neighbors in latent space are more meaningful than in
pixel / token space. The "Embedding exploration" demo
visualizes a learned latent.

## Layer

A parameterized transformation in a neural net. `linear`,
`embed`, `attention`, `rms_norm` are MLPL's core layers.
Activation layers (`tanh_layer`, `relu_layer`,
`softmax_layer`) carry no parameters.

## Layer norm

Normalize each input row to zero mean and unit variance,
then apply a learnable scale and shift. Distinct from batch
norm (which normalizes across the batch axis) and RMS norm
(which skips the mean step). MLPL ships `rms_norm(dim)`;
LayerNorm proper is deferred.

## Learning rate

The scalar that scales the gradient before each update step.
Too high: the loss diverges. Too low: training is glacial.
MLPL: `cosine_schedule` and `linear_warmup` produce
`LearningRate`-tagged scalars on schedule.

## Learning Rate Schedules

Strategies that change the learning rate during training.
Classics: linear warmup (ramp up early), cosine decay (smooth
ramp down), step decay (drop by a factor at fixed steps).
MLPL: `cosine_schedule(step, total, lr_min, lr_max)` and
`linear_warmup(step, warmup, lr)`.

## Linear (layer)

`y = x @ W + b`. A dense matrix multiply plus a bias. The
fundamental building block of MLPs and transformers.
Spelled `linear(in_dim, out_dim, seed)` in MLPL.

## LM (Language Model)

A model that predicts the next token given a sequence of
prior tokens. The "Tiny LM" and "Tiny LM Generate" demos
train a 1-layer transformer end-to-end on a small corpus.

## llm_call (builtin)

`llm_call(url, prompt, model)` POSTs to an Ollama-compatible
`/api/generate` endpoint and returns the model's completion
text as a `Value::Str` string. Saga 19 added this; CLI-only
in v0.19 (browser CORS / proxy story is a deferred saga).

## load / load_preloaded (builtins)

`load("rel.csv")` / `load("rel.txt")` reads through an
`Environment::data_dir` sandbox set by the terminal REPL's
`--data-dir` flag. `load_preloaded("name")` serves
compiled-in corpora for the web REPL where filesystem access
is unavailable. Both produce a string for `.txt` and a
DenseArray (with header autoparse) for `.csv`.

`load_preloaded("pets_tiny")` (Saga 29 step 003) returns a
`Value::Record` with three fields: `X` (a `DenseArray` of
shape `[200, 3, 64, 64]` with `[batch, channel, y, x]` axis
labels), `Y` (a `[200]` label vector; `0 = cat`, `1 = dog`),
and `names` (a `Value::StrList` of source filenames). The
fixture is shipped as pre-decoded u8 RGB bytes via
`include_bytes!` so the WASM REPL has it without any live
decoder.

## fetch_dataset (builtin)

`fetch_dataset(name)` (Saga 29 step 004, native-only via the
`image-io` Cargo feature) is the live counterpart to
`load_preloaded`. The v0.21 registry recognizes one name --
`"oxford_iiit_pet"` -- which downloads the upstream
~792 MB tarball to `$MLPL_DATA_DIR/oxford-iiit-pet/` on first
use, sha256-verifies against a pinned hash, untars to
`images/`, then runs the same decode + bilinear-resize +
normalize pipeline as `load_images` at the demo's 128x128
resolution. Returns the same record shape as
`load_preloaded("pets_tiny")` but with the full ~7393-image
count instead of 200. Pre-populated checkouts (the tarball
and extracted `images/` already on disk) skip HTTP entirely,
so a session that already ran the download from a prior
step pays no network cost. The WASM REPL gets a clean error
pointing at the preloaded fixture; image decoders are
deliberately not in the WASM dependency tree.

## patchify (builtin)

`patchify(x, P)` rearranges a `[B, C, H, W]` image batch into
`[B, N, P*P*C]` patch tokens (Saga 29 step 005). `P` is the
square patch side length; it must divide both `H` and `W`,
giving `N = (H/P) * (W/P)` patches per image. Each row of the
trailing axis is one patch flattened in channel-outer order:
the element at `(c, dy, dx)` lands at flat index
`c * P*P + dy * P + dx`. Patch traversal across `N` is
row-major: `n = i * (W/P) + j` for patch row `i` and column
`j`. Differentiable on the autograd tape: forward is a pure
re-arrangement, backward scatters the gradient back to image
space with the inverse indexing (no per-position accumulation
since every output element comes from exactly one input).

The named builtin exists for two reasons. First, the demo
reads cleanly (`tokens = patchify(images, 16)` instead of
three reshape + transpose + reshape calls). Second, the
autograd tape has one named op to lower, which simplifies the
backward implementation compared to lowering through a
general `permute` primitive that MLPL doesn't ship yet.

## concat (builtin)

`concat(a, b)` (Saga 13) joins two rank-0 or rank-1 arrays
into a 1-D vector; used by generation loops to append a
sampled token id to the growing sequence.

`concat(a, b, axis)` (Saga 29 step 005) is the axis-aware
extension. Both inputs must agree on every dim except `axis`,
where the sizes add. Initial release supports `axis` in
`{0, 1}` only; higher axes are a follow-up. Differentiable on
the tape: forward stacks data per the axis layout; backward
splits the upstream gradient at the seam (`left_size` along
`axis`) and delivers each half to its parent. The driving use
case is CLS-token prepending in ViT: `concat(cls, patches, 1)`
adds a learnable `[B, 1, D]` token to the front of a
`[B, N, D]` patch sequence so the classifier head can read off
the CLS row after attention.

## load_images (builtin)

`load_images(dir, [H, W])` (Saga 29 step 003, native-only via
the `image-io` Cargo feature) reads every PNG / JPEG under
`dir`, decodes via magic-byte dispatch to `png` /
`jpeg-decoder` (smaller dep footprint than `image-rs`),
bilinear-resizes to `(H, W)`, normalizes pixel bytes to f64
in `[-1, 1]`, and returns a `[N, 3, H, W]` `DenseArray` with
`[batch, channel, y, x]` axis labels. The WASM build raises a
clean error pointing users at the `pets_tiny` fixture
instead, since the WASM target deliberately excludes image
decoder dependencies.

## log (builtin)

Elementwise natural log: `log(x)` returns `ln(x)` for each
element. Used to bridge `Probability -> LogProbability` and
in numerical stability tricks for cross-entropy.

## LM benchmarks (BLEU, ROUGE, MMLU, HellaSwag)

Standard evaluation suites for language models. **BLEU**
and **ROUGE** are n-gram-overlap metrics for translation
and summarization (closer to 1 is better; both have known
limitations). **MMLU** (Massive Multitask Language
Understanding) is a 57-subject multiple-choice exam --
the standard "general knowledge" leaderboard.
**HellaSwag** tests commonsense sentence completion. None
ship as MLPL builtins; benchmark suites are downstream
application work. Glossary anchor for the names that
appear in LM evaluation papers.

## Logits

The unnormalized scores a classifier produces just before
the softmax. A `Logit` tag in v0.19 -- `cross_entropy`,
`sample`, and `top_k` expect them; passing `softmax`
output instead is the canonical double-softmax bug.

## LogProbability

The log of a probability. Numerically stabler than
multiplying probabilities; `log_softmax` (deferred to a
later saga) produces them.

## LoRA (Low-Rank Adaptation)

Parameter-efficient fine-tuning. Replace each `Linear` with
`y = x @ W + (alpha/rank) * x @ A @ B + b` where `A`, `B`
are trainable low-rank adapters and `W`, `b` are frozen.
MLPL: `lora(model, rank, alpha, seed)`.

## Loss

A scalar that summarizes how wrong a model's predictions
are. Minimization target for `adam` / `momentum_sgd`.
Auto-tagged `Loss(kind)` in v0.19.

## Loss Landscape

The high-dimensional surface defined by parameter values
mapped to loss values. Modern theories describe it as full
of saddle points, narrow valleys, and wide flat minima --
the latter generalize better. Tools to probe the landscape
(loss surface sharpness, Hessian eigenvalues, SAM) are
research targets.

## Lottery Ticket Hypothesis

Empirical claim that inside a randomly-initialized dense
network there exist sparse subnetworks ("winning tickets")
that, trained from scratch with their original initialization,
match the full network's accuracy. Drives interest in
pruning + retraining.

## Manifold Hypothesis

Real-world high-dimensional data (images, text, audio) lies
near a much-lower-dimensional manifold inside the ambient
space. Justifies dimensionality reduction (PCA, t-SNE, UMAP)
and explains why deep learning works at all -- the network
need only be expressive on the manifold, not the cube.

## map (deferred higher-order)

`map(:op, x)` -- elementwise apply of a unary BuiltinRef
across every element. Not shipped in v0.19; the natural
companion to `reduce(:op, x[, axis])` and an obvious follow-
up. For now compose with `reduce(:add, x * x)` or named
math primitives (`exp(x)`, `sigmoid(x)`).

## Mask

A `0 / 1` (or `0.0 / -inf`) array that nullifies positions in
a downstream op. Causal attention applies a lower-triangular
`-inf` mask before softmax so future positions get zero
probability. MLPL handles this internally for
`causal_attention`; explicit `batch_mask` is also returned by
`batch(x, size)` for short-batch padding.

## Matmul

Matrix multiplication. `matmul(A, B)` requires
`A.cols == B.rows`. The single most common operation in
neural-net forward passes.

## Mixed Precision

Storing weights in f32 for stability while running matmuls in
f16 / bf16 for speed and memory. Standard on modern GPUs.
MLPL stores everything in f64 today; mixed precision is
deferred until a dtype layer ships.

## mean (builtin)

`mean(x)` returns the arithmetic mean of all elements as a
scalar. Distinct from `reduce_add(x) / shape(x)` only in
that it ignores axis arguments today (always full reduction).

## MLP (Multi-Layer Perceptron)

A stack of `linear` + activation layers. The "Tiny MLP" and
"Moons MLP" demos show 2-layer MLPs; the Model DSL writes
the same model as `chain(linear(2, 8, 0), tanh_layer(),
linear(8, 2, 1))`.

## Mode Collapse

When a generative model (typically a GAN) produces only a
narrow slice of the data distribution, ignoring whole
clusters. The classic GAN failure mode; harder to define for
diffusion or autoregressive models. Out of MLPL's current
scope.

## Model

A runtime callable composed from one or more layers. MLPL's
`Value::Model` wraps a `ModelSpec` tree built by `linear`,
`chain`, `residual`, `attention`, etc. Apply with
`apply(model, X)`.

## MoE (Mixture of Experts)

Replace a single feed-forward layer with `k` parallel "expert"
sub-networks plus a "router" that sends each token to one or
two of them. Increases parameter count without proportional
inference cost, since only a fraction of experts run per
token. Out of MLPL's current scope.

## Momentum SGD

Gradient descent with a running velocity. `momentum_sgd(loss,
params, lr, beta)` accumulates a momentum vector that
smooths out gradient noise.

## MSE (Mean Squared Error)

Regression loss: `mean((pred - target)^2)`. Used when the
target is continuous rather than a discrete class.

## Multi-head attention

Splits a `d_model`-wide tensor into `h` heads of width
`d_model / h`, runs attention on each in parallel, then
concatenates the results. Each head learns its own
projection, capturing different relationships in the input.
MLPL: `attention(d_model, heads, seed)` -- `heads` controls
this split.

## :models / :tokenizers (REPL commands)

`:models` lists every bound `Value::Model` with its layer
tree summary; `:tokenizers` does the same for tokenizer
bindings. Sibling commands to `:vars` for the non-array
namespaces.

## One-hot encoding

Converting an integer class index to a vector with 1.0 at
that index and 0.0 elsewhere. MLPL: `one_hot(labels,
num_classes)`.

## Oxford-IIIT Pet dataset

7,393 photographs of cats and dogs (12 cat breeds + 25 dog
breeds, ~200 images per breed), released by the Visual
Geometry Group at Oxford. Standard cat-vs-dog classification
benchmark with breed-level subclasses. MLPL uses it as the
training set for the Saga 29 Vision Transformer demos. The
filename convention encodes the class: capitalized prefix =
cat breed (`Abyssinian_1.jpg`), lowercase prefix = dog breed
(`beagle_3.jpg`). The full ~792 MB tarball lives in a
gitignored `data/oxford-iiit-pet/` checkout; the
`pets_tiny` preloaded fixture committed under
`crates/mlpl-eval/data/pets_tiny.bin` is the 200-image
(100 cat + 100 dog) subset used by the WASM REPL demos.

## OOD Inputs (Out-of-Distribution)

Inputs that fall outside the distribution the model was
trained on. Models tend to be silently overconfident on OOD,
not flagged "I don't know". Detection is an active research
area; deferred in MLPL.

## Online vs offline distillation

**Offline distillation:** train the teacher to convergence
first, then train the student against the frozen teacher's
logits. The standard pipeline; reproducible and easy to
debug.
**Online distillation:** train teacher and student
together, often with the teacher's logits computed in the
same forward pass. Cheaper end-to-end but the moving
target makes debugging harder. **Deferred** in MLPL: both
require the `kl_div` / `soft_targets` builtins flagged as
Module 11 blockers.

## Optimization vs Generalization

Optimization concerns minimizing training loss; generalization
concerns minimizing test loss. They diverge: a sufficiently
expressive model can memorize the training set (zero
optimization gap, large generalization gap). Implicit and
explicit regularization closes the gap.

## Optimizer

The procedure that updates parameters from gradients.
`adam` and `momentum_sgd` are MLPL's two; both keep
per-parameter state on the environment.

## Overfitting / Underfitting

Overfitting: the model fits training data better than the
underlying function (high variance, low bias). Underfitting:
the model lacks the capacity to fit at all (high bias, low
variance). Classic diagnosis: train loss low + val loss high
= overfitting; both high = underfitting; both low =
well-tuned.

## pairwise_sqdist (builtin)

`pairwise_sqdist(X)` returns the `[N, N]` symmetric matrix
of pairwise squared Euclidean distances between every pair
of rows in `X`. Building block for `knn`, k-means, and
embedding-cluster inspection.

## param / tensor (constructors)

`param[d0, d1, ...]` allocates a trainable leaf tensor that
the autograd tape tracks; `tensor[d0, d1, ...]` allocates a
fixed (non-trainable) leaf. Both auto-bind to the assignment
target's name. Auto-tagged `Weight` / `Bias` per the
shape-and-position heuristics in Saga 23.

## Padding

Filling a short input out to a fixed length so a batch can be
rectangular. MLPL: `batch(x, size)` zero-pads the trailing
short batch and returns a `batch_mask` so downstream ops can
ignore the padded positions.

## Parameter

A trainable tensor. Bound via `param[shape]` or by a Model
DSL constructor. The `params(model)` walker returns the
flat list of names a model owns.

## PCA (Principal Component Analysis)

A linear projection that finds the directions of greatest
variance in a dataset. MLPL: `pca(X, k)` returns the
top-k projection `[N, k]`. The "PCA via Power Iteration"
demo writes it out by hand.

## perturb_params (builtin)

`perturb_params(m, family, sigma, seed)` walks `m`'s param
tree, filters by `family` (`"all_layers"`, `"attention_only"`,
`"mlp_only"`, `"embed_and_head"`), and adds `sigma * randn(seed
+ i, shape)` to each matching param in place. Saga 20's
weight-perturbation ensembling pattern.

## Perceptron

A single linear layer plus a step / sigmoid activation -- the
1958 ancestor of every neural network. Limited to linearly-
separable problems alone; the famous Minsky-Papert XOR
critique drove research into multi-layer networks (MLPs).

## Perplexity

The exponentiated cross-entropy of a language model on a
held-out corpus: `exp(cross_entropy_loss)`. Standard LM
evaluation metric. Lower is better. MLPL doesn't ship a
dedicated `perplexity` builtin; `exp(cross_entropy(...))`
gives the same number.

## Positional encoding

Information added to each token's embedding so the model can
distinguish position in a sequence (attention is otherwise
permutation-invariant). MLPL: `sinusoidal_encoding(T,
d_model)` produces a deterministic `[T, d_model]` table to
add to embedded tokens.

## Precision vs Recall

For a binary classifier: precision = `TP / (TP + FP)` (of
flagged positives, what fraction were truly positive); recall
= `TP / (TP + FN)` (of truly-positive examples, what fraction
did we flag). Tunable via the decision threshold; F1 is the
harmonic mean. ROC / AUC summarizes the precision-recall
tradeoff over all thresholds.

## Preference Learning

Training with pairwise rankings ("output A is preferred to
output B") instead of absolute labels. The DPO and RLHF
families. Out of MLPL's current scope.

## Pipeline Parallel Training

Split the layers of a deep model across devices: device 0
holds layers 1-3, device 1 holds 4-6, etc. Inputs flow
through the device chain (forward), then gradients flow
back (backward). To keep all devices busy, micro-batches
are pipelined so each device works on a different
micro-batch at any moment. Distinct from data parallel
(replicate, split batch) and tensor parallel (split a
layer's weights). **Deferred** in MLPL.

## Pretraining / fine-tuning

Pretraining: train a base model on a large generic corpus.
Fine-tuning: continue training on a smaller, task-specific
dataset, often with a smaller subset of parameters trainable
(see LoRA). MLPL's "Tiny LM" demos pretrain; the "LoRA
Fine-Tuning" lesson fine-tunes.

## pow (builtin)

Elementwise power: `pow(a, b)` raises each element of `a` to
each element of `b`, broadcasting in the usual way. `pow(x,
2.0)` is the canonical squared-error component.

## power iteration

Numerical method for the dominant eigenvector: repeatedly
multiply `Cov * v` and renormalize until `v` stops changing.
The "PCA via Power Iteration" demo writes it out by hand;
the `pca(X, k)` builtin uses the same idea internally with
Gram-Schmidt deflation for the top-k.

## Probability

A non-negative scalar that, with siblings, sums to 1.
Auto-tagged `Probability` in v0.19; produced by `softmax`
and `sigmoid`.

## Projection (matrix)

A learned linear map applied as `Y = matmul(X, W)` where `W
: [d_in, d_out]`. In attention, three projections (Wq, Wk,
Wv) turn each token into Query, Key, Value vectors of width
`d_model`; a fourth (Wo) recombines per-head outputs after
concatenation. In MLPL these are `param[d_in, d_out]`
tensors trained by gradient descent through `apply` /
`attention`.

## Prompt Injection

Adversarial inputs that overwrite the system prompt or steer
the model to leak instructions / tools. The web equivalent
of SQL injection. MLPL exposes `llm_call` but the security
posture is the user's responsibility.

## Prompt masking

In supervised instruction tuning, zero out the loss on
prompt tokens so the gradient only flows from the
response. Without it the model would also learn to
predict the user's instruction text, which is wasted
capacity. Implemented as a per-position mask multiplied
into the per-token cross-entropy before averaging.
**Deferred** in MLPL: requires per-position loss masking,
which `cross_entropy(logits, targets)` does not expose.

## Prompting

Crafting the input string that conditions an LLM's output.
The lever everything else (few-shot, chain of thought,
RAG context) attaches to. No dedicated MLPL builtins; just
strings passed to `llm_call(url, prompt, model)`.

## QLoRA

Quantization-aware LoRA fine-tuning: load the base model
in int4 (or other low-precision format) with weights
frozen, and train only the LoRA adapter weights in higher
precision (typically bfloat16). Lets fine-tuning fit on
far smaller GPUs than the base model would otherwise
allow. Pairs naturally with the chat-template / SFT
workflow. **Deferred** in MLPL: the LoRA path exists
today (Saga 17) but `quantize` does not.

## Quantization

Storing weights in low-precision integer formats (int8, int4)
for memory and speed at modest accuracy cost. Often combined
with LoRA in QLoRA workflows. MLPL stores everything in f64;
quantization is a deferred follow-up to Saga 15.

## Query (Q)

One of the three projections in attention. Each token emits
a query asking "what am I looking for?"; the dot product `Q
@ K^T` measures how strongly each query matches each key. In
MLPL: `Q = matmul(X, Wq)` where `Wq` is `[d_model, d_model]`
for single-head or `[d_model, d_k]` per head. Paired with
Key (K) and Value (V).

## RAG (Retrieval-Augmented Generation)

Fetch relevant documents from a corpus at query time, prepend
them to the prompt, and let an LLM answer over the retrieved
context. Reduces hallucination and lets you cite. MLPL ships
`pairwise_sqdist` / `knn` for similarity search; full RAG
pipelines are a deferred saga.

## randn / random (builtins)

`randn(seed, [shape...])` returns a standard-normal
sample (mean 0, variance 1) with the given shape. `random(
seed, [shape...])` returns uniform `[0, 1)` samples. Both
deterministic given the same seed.

## Rank (of a tensor)

The number of dimensions of a tensor. A scalar has rank 0; a
vector rank 1; a matrix rank 2; a `[batch, time, dim]`
tensor rank 3. MLPL: `rank(x)` returns the count;
`shape(x)` returns the dim sizes. Distinct from "rank" in
linear algebra (the dimension of a matrix's column span)
and from "low-rank" in LoRA (the small inner dimension of
the adapter matrices A and B).

## reduce (builtin)

Higher-order reduction: `reduce(:op, x[, axis])` applies the
binop named by a `BuiltinRef` to every element of `x` (or
along `axis`), starting from the op's identity. Curated set:
`:add` (== `:+`), `:mul` (== `:*`), `:min`, `:max`, `:and`,
`:or`. Examples: `reduce(:max, v)`, `reduce(:add, M, 1)`,
`f = :max; reduce(f, v)`. Subsumes the older fixed-name
`reduce_add` / `reduce_mul`. See also: `dot product`, `mean`,
`argmax`.

## reduce_add / reduce_mul (builtins)

`reduce_add(x[, axis])` is sum reduction; `reduce_mul` is
product reduction. Equivalent to `reduce(:add, x[, axis])`
and `reduce(:mul, x[, axis])`; kept as direct shorthands.

## Random Forest

An ensemble of Decision Trees, each trained on a bootstrap
sample of the data with a random feature subset considered
at every split. Predictions are averaged (regression) or
voted (classification). Reduces variance vs a single deep
tree, harder to overfit, still interpretable feature-by-
feature. Distinct from Boosting (sequential trees on
residuals, e.g. XGBoost). **Deferred** in MLPL.

## Record

A structured value built from `{name1: expr1, name2: expr2}`
record-literal syntax (Saga 29 step 001). Field access is
`r.name`. Distinct from `{ stmt; }` blocks, which only appear
after the `repeat` / `train` / `for` / `experiment` / `device`
keywords -- at any other position `{` opens a record. Field
keys are idents, must be unique within a literal (duplicates
parse-error), and iterate alphabetically (BTreeMap) regardless
of source order. Field values can be any value (nested records,
arrays, strings, models, ...). Unknown-field access returns
`FieldNotFound { requested, available }` listing the valid
keys; field access on a non-record returns
`FieldOnNonRecord { receiver_kind, field }`. Use case: the
Saga 29 Vision Transformer track wants
`load_preloaded("pets_tiny")` to return
`{X: [200, 3, 64, 64], Y: [200], names: [str]}` -- one builtin,
three logical outputs, no positional-tuple awkwardness.

Out of scope for the initial step: record destructuring in
let-bindings (`let {X, Y} = r`), record-update / spread syntax
(`{..r, X: new_x}`), pattern matching on records. Each is a
separate follow-up if a use case appears.

## Regularization

Anything that constrains the model away from overfitting:
weight decay, dropout, early stopping, KL terms, parameter
norms, etc. MLPL doesn't ship a dedicated regularization
surface today; you can compose terms manually in the loss.

## ReLU

Rectified Linear Unit: `max(x, 0)`. The most common
activation in modern deep learning. MLPL: `relu_layer()`
or the math builtin path.

## Replay Buffer

A circular buffer of past experiences (state, action, reward,
next state) that an off-policy RL algorithm samples from
during training. Out of MLPL's current scope; RL builtins
are a future saga.

## Representation Learning

The umbrella term for learning useful internal features
without explicit feature engineering. Self-supervised
pretraining, autoencoders, contrastive learning all fall
under it. Saga 16's embedding-visualization tools poke at
representations; full self-supervised pretraining is a
deferred saga.

## repeat block (language keyword)

`repeat N { body }` runs the body `N` times with no per-
iteration index binding. Ancestor of `train { ... }` (which
DOES bind `step` and capture loss). Use `repeat` for
iterative algorithms (k-means, power iteration, MLP forward
+ backward demo) where you want a counted loop without the
training-specific bookkeeping.

## reshape (builtin)

`reshape(x, [d0, d1, ...])` returns a view of `x` with the
given dim sizes. Total element count must match; otherwise
`ShapeMismatch`. Clears axis labels (semantic identity is
lost on shape reflow); `reshape_labeled(x, dims, labels)`
preserves them by re-stating labels explicitly. Note: also
clears Saga 23 ValueTags, since the result no longer
represents the same domain.

## Residual

`y = x + f(x)`. A skip connection that lets gradients flow
through deep stacks. MLPL: `residual(inner_model)`.

## Reward Hacking

When an RL agent finds a strategy that maximizes the reward
signal without solving the intended task -- exploiting bugs
in the reward function or environment. Goodhart's Law applied
to RL. Out of MLPL's current scope.

## ResNet (Residual Network)

A deep CNN (or transformer) where every block computes
`y = x + f(x)` instead of `y = f(x)`, so the gradient has
a clean path through the identity bypass. Solved the
"deeper means harder to train" problem and unlocked
50-layer-plus networks. The `residual(...)` builder in
MLPL's model DSL is exactly this pattern; the Encoder
Block / Decoder Block lessons use it on every sub-block.

## RLHF (Reinforcement Learning from Human Feedback)

Three-stage training: (1) supervised fine-tune a base model,
(2) train a reward model on human preferences, (3) RL the
policy against the reward model. The standard recipe behind
modern instruction-tuned LLMs. Out of MLPL's current scope.

## RMS Norm

Root-mean-square normalization: `x / sqrt(mean(x^2) + eps)`.
A simpler alternative to LayerNorm. MLPL: `rms_norm(dim)`.

## RNN / LSTM / GRU

Recurrent network families that process sequences one token
at a time and pass a hidden state forward. Largely displaced
by transformers for language tasks. MLPL does not ship
recurrent layers in v0.19; deferred.

## ROC / AUC

Receiver Operating Characteristic curve plots true-positive
rate vs false-positive rate as the decision threshold
sweeps; AUC is the area under the curve, a single-number
summary of the classifier's ranking quality (`0.5` = random,
`1.0` = perfect). MLPL doesn't ship ROC / AUC builtins;
`confusion_matrix` covers single-threshold evaluation.

## RoPE (Rotary Positional Embeddings)

A positional encoding scheme that rotates each query / key
pair in a 2-D plane proportional to position, so attention
naturally prefers nearby tokens and extrapolates better
beyond training context length. MLPL ships only sinusoidal
positional encoding today; RoPE is deferred.

## Sampling

Drawing a random outcome from a probability distribution.
Multinomial sampling from logits: `sample(logits,
temperature, seed)`.

## Scaled dot-product attention

The canonical attention formula: `softmax(Q @ K^T / sqrt(d_k),
1) @ V`. The `sqrt(d_k)` divisor keeps the score variance
bounded as the key dimension grows so softmax doesn't
saturate into one-hot. Demos: "Attention Pattern" (heatmap
of weights only) and "Self-Attention from Scratch" (full
pipeline including `weights @ V`). Multi-head attention runs
this formula in parallel on `d_k = d_model / heads`-wide
slabs, then concatenates the per-head outputs.

## Scaling Laws

Empirical regularities relating model performance to model
parameters, dataset tokens, and compute budget -- typically
power laws with predictable exponents. Drove the "just make
it bigger" era of foundation models. The Compute-Optimality
Hypothesis (Chinchilla scaling) is a refinement: at a fixed
compute budget, smaller models trained on more data beat
bigger models trained on less.

## Self-attention

Attention where the queries, keys, and values all come from
the same input sequence -- each position scores against every
other position to produce a context-aware representation.
The diagonal-heavy attention map is the visual signature.
MLPL: `attention(d_model, heads, seed)` is self-attention by
default; `causal_attention(...)` adds the lower-triangular
mask used in language models. The "Attention Pattern" demo's
second pass renders the diagonal pattern from `Q @ Q^T`.

## Self-play

A training regime where an agent plays against itself,
generating its own training signal -- AlphaZero / chess
engines / self-improving game players. Distinct from
human-supervised learning because no fixed labels exist;
the only ground truth is "did this strategy beat the other
copy of yourself?". Modern LLM post-training also borrows
the pattern (self-rewarding models, debate, constitutional
AI). **Deferred** in MLPL: requires environment + reward
+ policy-update primitives that do not ship today.

## Self-supervised learning

Training on (input, derived-label) pairs where the label
is computed from the input itself rather than supplied by
a human. Next-token prediction is the canonical example:
the label for token t is just token t+1. MLPL: the
"Tiny LM" demo's `shift_pairs_x(ids, block)` /
`shift_pairs_y(ids, block)` pairing is a self-supervised
setup -- no human labels touched the model. Most of modern
LLM pretraining is self-supervised.

## scan (deferred higher-order)

`scan(:op, x)` -- the cumulative version of `reduce`. Returns
a same-shape array where each entry is the reduction over the
prefix up to that point. Standard for cumulative sum / running
max / prefix product. Not in v0.19; obvious follow-up to
`reduce`.

## scatter (builtin)

`scatter(buf, idx, value)` returns a copy of a rank-1 buffer
with the entry at `idx` replaced by `value`. Saga 20's neural-
thicket loop uses it to write each variant's loss into a
flat `[16]` accumulator before reshaping into the heatmap.

## scatter_labeled (builtin)

`scatter_labeled(points, labels)` renders `[N, 2]` points
colored by integer class labels as an SVG scatter plot.
Stable color palette across calls so multiple runs are
visually comparable.

## Sequence

An ordered list of tokens. Inputs to language models are
`[batch, sequence_length]` integer arrays; outputs are
`[batch, sequence_length, vocab]` logit tensors.

## Shortcut Learning / Spurious Correlations

When the model latches onto a feature that correlates with
the label in training but not in deployment -- the famous
"husky vs wolf" paper where the model learned snow, not
wolves. Mitigations: data augmentation, careful curation,
causal-feature-aware training.

## shape / rank (builtins)

`shape(x)` returns the dim-size vector of an array (e.g.
`[2, 3]` for a 2x3 matrix). `rank(x)` returns the number of
dims (a single scalar). Together they describe an array's
structural type. Shape-mismatch errors at runtime cite the
`shape(x)` of the offending operand; named-axis arrays
render as `[batch=4, vocab=8]` instead of `[4, 8]`.

## shift_pairs_x / shift_pairs_y (builtins)

`shift_pairs_x(ids, block)` and `shift_pairs_y(ids, block)`
take a flat token-id array and produce next-token training
pairs of length `block`. `_x` rows are the input
sub-sequences; `_y` rows are the corresponding target
sub-sequences (input shifted by one). The Tiny LM demo uses
this to build training data without a custom loop.

## shuffle (builtin)

`shuffle(x, seed)` returns a Fisher-Yates row permutation of
a rank-2 array, deterministic given `seed`. Standard for
randomizing dataset order before batching.

## Sigmoid

`1 / (1 + exp(-x))`. Squashes any real number to `[0, 1]`.
The classic binary-classification activation. Auto-tags
its output as `Probability`.

## Sinkhorn normalization

An iterative procedure that turns a non-negative matrix into
a doubly-stochastic one (rows AND columns sum to 1) by
alternating row and column scaling. The continuous relaxation
of optimal transport / matching problems. MLPL does not ship
`sinkhorn_normalize` as a v0.19 builtin; it is on the
research3.txt wishlist alongside the other normalization
families.

## Softmax

Normalizes a vector of scores into probabilities that sum
to 1: `exp(x_i) / sum(exp(x_j))`. The standard bridge from
Logit to Probability. MLPL: `softmax(x, axis)` -- always
pass an axis; the implementation max-subtracts before exp
for numerical stability. Auto-tags its output as
`Probability` so `cross_entropy(softmax(...), Y)` raises a
TypeMismatch at the call site (the canonical double-softmax
bug).

## Soft targets

The output of a softmax over logits divided by a
*temperature* > 1: `soft_targets = softmax(logits /
temperature, axis)`. Temperature spreads probability mass
across more classes, so the targets carry richer
information than a one-hot label about which alternatives
the teacher considered. Training a student against these
softer targets via KL divergence is the core of knowledge
distillation. **Deferred** in MLPL: build by hand from
`softmax(logits / temperature, axis)`; a `soft_targets`
helper is on the Module 11 roadmap.

## Sparse Activation

Architectures where only a small fraction of parameters are
active per input (MoE routing, top-k attention, lottery-
ticket subnetworks). Trades dense matmul efficiency for
parameter efficiency. Out of MLPL's current scope.

## Speculative Decoding

Use a small "draft" model to generate `k` candidate tokens
ahead, then have the large model verify them in one parallel
forward pass. Accepts the longest matching prefix. Speeds up
LLM inference 2-3x with no quality loss. Out of MLPL's
current scope.

## State Space Models / Mamba

A non-attention sequence-modeling family that runs in
`O(seq_len)` time using a learned linear recurrence with
selective gating. Trades all-to-all attention for linear
scaling. MLPL ships transformer + MLP families only; SSMs
are deferred.

## sqrt (builtin)

Elementwise square root: `sqrt(x)`. Used in attention
scoring (`/ sqrt(d_k)`), RMS norm, and PCA's Gram-Schmidt
normalization. Negative inputs produce NaN; the runtime does
not raise.

## Step

One optimizer update. The training loop runs `train N { body
}` for `N` steps; each step computes a forward pass, a
backward pass, and one parameter update. Distinct from
"epoch" (a full dataset pass).

## String list

A `Value::StrList { items: Vec<String> }` value built from a
`["a", "b", "c"]` literal whose every element evaluates to a
string (Saga 29 step 002). The same `[...]` surface syntax
dispatches on element kind: all strings -> `StrList`; all
numbers -> the existing `DenseArray` numeric path; mixed
kinds -> `EvalError::MixedArrayLitElements { kinds }` so the
user sees which positions disagreed. Empty `[]` continues to
produce an empty `DenseArray` for back-compat.

Use case: the Saga 29 Vision Transformer track wants
`load_preloaded("pets_tiny")` to return
`{X: [200, 3, 64, 64], Y: [200], names: ["Abyssinian_1.jpg",
"beagle_3.jpg", ...]}` -- one record value with three logical
outputs, including a per-image basename list, without
shoehorning the names into an index-keyed sub-record.

Today's accessors are minimal: `list_len(xs)` returns the
length as a scalar `DenseArray`. Indexing (`names[i]` or
`index(names, i)`) and iteration are deferred to a follow-up
step once a concrete demo needs them.

## Stop gradient / detach

Severing the autograd tape so gradients do not flow through
a value. Useful for target networks, EMA teachers, and
"freeze the encoder" patterns. MLPL does not ship
`stop_gradient` / `detach` as a builtin in v0.19;
`freeze(model)` covers the optimizer-side equivalent for
parameters.

## Superposition

A mechanistic-interpretability finding: neural networks pack
many more "concept directions" into their hidden activations
than there are neurons, by representing them as overlapping
directions in feature space. Explains why a single neuron
rarely encodes one clean concept and motivates dictionary-
learning-style decomposition.

## :tags / :untag (REPL commands)

Saga 23 v0.19 typed-value introspection. `:tags` lists every
binding with an attached `ValueTag` sorted alphabetically,
showing the tag's display form (e.g. `Probability`,
`Loss(CrossEntropy)`, `Weight(layer=linear_0, name=W)`).
`:untag <name>` clears the auto-tag from a binding when the
auto-tagger guessed wrong.

## tanh_layer / relu_layer / softmax_layer (builtins)

Parameter-free activation layers wrappable in a `chain(...)`.
`tanh_layer()` / `relu_layer()` / `softmax_layer()` apply
their respective elementwise functions. Distinct from the
math primitives (`tanh_fn`, `sigmoid`) in that layers can
participate in `apply(model, X)` and Saga 23's structural-
tail tagging.

## Supervised learning

The classical paradigm: training on (input, label) pairs
where a human (or curation pipeline) supplied the labels.
The model minimizes a per-example loss against the label
-- cross_entropy for classification, MSE for regression.
MLPL: every classifier demo (Logistic Regression, Tiny
MLP, Moons MLP, Softmax Classifier) is supervised. Cheap
and well-understood, but bounded by the supply of
labeled data.

## SVM (Support Vector Machine)

A binary classifier that finds the maximum-margin
hyperplane separating two classes. The kernel trick lets
it implicitly use higher-dimensional spaces (RBF,
polynomial) so non-linearly-separable data becomes
separable in a transformed space. The dominant pre-deep-
learning classifier on small / tabular tasks; mostly
historical now. **Deferred** in MLPL: needs a quadratic-
program solver (or SMO algorithm).

## System prompt

The leading text in a chat-formatted prompt that sets the
model's persona, capabilities, and constraints, before the
user turn begins. Different from instruction prompts: the
system prompt is set by the application, not the user, and
is typically wrapped in a distinct role marker by the chat
template. Adversarial inputs that try to override it are
called prompt injection. MLPL: not a first-class concept;
build via `llm_call` with a manually-formatted prompt
string.

## Tanh

`(exp(x) - exp(-x)) / (exp(x) + exp(-x))`. Squashes any real
number to `[-1, 1]`. Older alternative to ReLU; still
useful in small MLPs. MLPL: `tanh_layer()` or `tanh_fn`.

## Temperature

The scalar that controls how peaky a softmax is. Low
temperature (~0.1) concentrates mass on the top score;
high temperature (~2.0) flattens the distribution.

## Tensor

An N-dimensional array. MLPL's `DenseArray` is the storage;
`Shape` and `LabeledShape` carry the structural type.

## Tensor Parallel Training

Split a single layer's weights across multiple devices --
e.g. each device holds a column slab of a large linear
layer's weight matrix; activations are sharded along the
appropriate axis. Every forward step requires all-reduce
or all-gather across devices. Used for layers too big to
fit on one GPU. Distinct from data parallel (replicate
the whole model) and pipeline parallel (split layers
across devices). **Deferred** in MLPL.

## Test set

A final held-out slice of the dataset, never seen during
training or validation, used once for the report number.
Distinct from the validation set (which IS used during
development to tune hyperparameters / pick checkpoints).

## Token

The atomic unit of a sequence. After BPE tokenization, an
input string becomes an integer-id sequence; the model
embeds each id into a `d_model`-dimensional vector.

## Tokenizer

The bidirectional mapping between strings and integer-id
sequences. MLPL: `train_bpe(corpus, vocab_size, seed)`
returns a tokenizer; `apply_tokenizer(tok, text)` and
`decode(tok, ids)` cross the boundary.

## Tool Use

Patterns where an LLM produces structured output that
triggers an external function (web search, calculator, code
execution), then folds the result back into its context.
MLPL ships `llm_call` as the language-level hook; full tool-
calling protocols are deferred.

## Top-k

Restrict a logit vector to its k largest entries (zero out
the rest) before sampling. Reduces tail noise in
text generation. MLPL: `top_k(logits, k)`.

## tokenize_bytes / decode_bytes (builtins)

`tokenize_bytes(text)` returns a flat array of byte values
(`0..256`); `decode_bytes(ids)` reverses it. The simplest
tokenizer: byte-level, no merges, no vocabulary file. Useful
for tiny pedagogical demos before BPE complexity.

## Train (loop)

`train N { body }` runs the body N times, binding the loop
index to `step` and capturing the body's final scalar into
`last_losses` for plotting. Replaces the
`repeat N { adam(...); record_loss }` recipe.

## Train / Validation / Test Split

The standard three-way partition: train (gradient updates),
validation (model selection / hyperparameter tuning), test
(final report). Held strictly disjoint. MLPL: `split(x,
frac, seed)` and `val_split(x, frac, seed)` slice train
vs val; the test set is whatever you keep untouched.

## Transfer Learning

Reusing a model trained on one task as a starting point for
another. Includes pretraining + fine-tuning (LoRA, full
fine-tune), feature extraction (use frozen embeddings), and
domain adaptation. The economic engine of modern deep
learning.

## Transformer

A sequence model built from stacked attention + feed-forward
blocks with residual connections and normalization. The
"Tiny LM" demos build a 1-layer transformer.

## Transformer block

One layer of a transformer: pre-norm -> self-attention ->
residual; pre-norm -> feedforward -> residual. The unit you
stack to make deep models (12-100+ blocks for production-
scale transformers; MLPL demos use 1). MLPL builds one with
`chain(residual(chain(rms_norm(d), attention(d, h, s))),
residual(chain(rms_norm(d), linear(d, d_ff, s2),
relu_layer(), linear(d_ff, d, s3))))`.

## take (builtin)

`take(x, axis, idx)` drops one axis at a single integer index
(Saga 29 step 007). For a rank-`r` input, the result has rank
`r - 1`; per-axis labels carry through except for the dropped
one. `axis` and `idx` are eager-evaluated scalars. Driving use
case is per-image extraction in the ViT trained demo:
`take(load_preloaded("pets_tiny").X, 0, i)` returns one
image's `[3, 64, 64]` from the 200-image batch; `take(seq, 1,
0)` pulls the CLS row out of a `[B, 17, 128]` post-attention
activation.

Differentiable on the autograd tape. The backward scatters
the upstream gradient into a zero-filled array of the
parent's shape, placing the slice's gradient at
`axis = idx`. Equivalent to PyTorch's `index_select` for a
single index. Multi-index `gather` and slice ranges are
deliberate followups.

## Transpose

Swap rows and columns. `transpose(x)` reverses axis order;
preserves Logit / Probability / etc. tags.

## t-SNE

A non-linear projection that emphasizes local neighborhoods
when reducing high-dimensional points to 2-D. MLPL:
`tsne(X, perplexity, iters, seed)`. See "Embedding
exploration" lesson.

## U-Net

A convolutional encoder-decoder with skip connections at
every resolution: the contracting path downsamples, the
expanding path upsamples, and same-resolution feature maps
from the encoder are concatenated into the decoder. The
"U" comes from drawing the architecture diagram in the
shape of the letter. Originally for biomedical image
segmentation; now the standard backbone for diffusion
models. **Deferred** in MLPL: needs `conv2d` plus
upsampling primitives.

## Uncertainty Estimation

Asking "how confident is the model in this prediction" --
distinct from the softmax's calibration. Approaches: model
ensembles, Bayesian neural nets, Monte-Carlo dropout, deep
evidential learning. MLPL's "Neural Thicket" ensemble demo
is a tiny taste; full uncertainty surface is deferred.

## unfreeze (builtin)

`unfreeze(m)` is the inverse of `freeze(m)` -- removes every
param of `m` from the env's frozen set so subsequent
`adam` / `momentum_sgd` updates can move them again. Saga 15
ships both as the LoRA freeze / unfreeze pair.

## Universal Approximation

Theorem: a feed-forward network with a single hidden layer
can approximate any continuous function on a compact domain
to arbitrary precision, given enough hidden units. Existence
result, not a learnability claim. Justifies why neural nets
are not architecturally limited; says nothing about whether
gradient descent will find the right weights.

## Unsupervised learning

Training without labels: the model discovers structure in
the data itself. Clustering, dimensionality reduction,
density estimation. MLPL: the K-Means demo, the PCA demo,
and t-SNE (`tsne(X, perp, iters, seed)`) are unsupervised
-- they group / project points using only geometry.
Self-supervised learning is a closely-related modern
variant where labels are *derived* from the input rather
than absent.

## VAE (Variational Autoencoder)

An autoencoder where the latent space is regularized to
match a prior distribution (typically Gaussian). The encoder
outputs `(mean, std)`; the decoder samples from
`Gaussian(mean, std)`. Trained with reconstruction + KL-
divergence terms. Saga 24 plans first-class
`Distribution` support; VAE demos follow once
distributions ship.

## :vars (REPL command)

`:vars` lists every bound array variable with its shape
(labeled if any axes are named) and Saga 23 ValueTag if any.
Trainable params are flagged `[param]`; frozen params show
in `:wsid`'s frozen-count.

## Validation set

A held-out slice of the dataset used to measure generalization
during training without leaking into the gradient. MLPL:
`split(x, frac, seed)` and `val_split(x, frac, seed)` slice
disjoint chunks; the `experiment "name" { body }` block
captures `_metric`-suffixed scalars per run.

## Value (V)

One of the three projections in attention. Each token emits
a value that gets mixed into other tokens' outputs in
proportion to the softmax weights: `out = weights @ V` where
each output row is a weighted average of the V rows. In
MLPL: `V = matmul(X, Wv)` where `Wv` is `[d_model, d_model]`
for single-head or `[d_model, d_k]` per head. Paired with
Query (Q) and Key (K).

## ViT (Vision Transformer)

Apply a transformer directly to images by splitting the
image into fixed-size patches (e.g. 16x16 pixels),
flattening each patch into a vector, and treating the
sequence of patch vectors as tokens (with positional
embeddings). Showed that the inductive biases of CNNs are
not strictly required at scale -- pure attention works
on images too. Foundation for modern vision-language
models. **Deferred** in MLPL: needs image inputs + patch-
embed plumbing.

## VLM (Vision-Language Models)

Models that take both image and text inputs (e.g. CLIP,
LLaVA, GPT-4V). Out of MLPL's current scope -- the language
core is text + tabular today; vision pipelines are a
follow-up saga.

## Vocabulary

The set of tokens a model can input or output, indexed
`[0, V)`. After `train_bpe`, the vocab size is the cap you
set; the embedding `[V, d_model]` and final projection
`[d_model, V]` use it as their outer dimension.

## Warmup

Increase the learning rate linearly from a small value to the
target over the first few hundred steps so the optimizer
state stabilizes before the network sees full-sized updates.
MLPL: `linear_warmup(step, warmup_steps, target_lr)`.

## Weight

The trainable matrix in a `Linear` layer (or a tile of one
in `Attention`). Auto-tagged `Weight(layer, name)` in
v0.19.

## :wsid (REPL command)

`:wsid` (workspace ID) prints summary counts: variables,
trainable parameters, frozen parameters, models, tokenizers,
optimizer slots, experiment records. Inspired by APL's
`)WSID`. The first command to run when you reopen a
session and want a quick sense of state.

## Weight Decay

A regularization technique that shrinks weights toward zero
on each update step (L2 penalty). Often baked into the
optimizer (`AdamW`). MLPL's `adam` uses no decay; weight
decay is deferred.

## Weight Initialization

The scheme that sets parameter values before training: zeros
(biases), small Gaussian (general weights), Xavier / Glorot
(scaled by fan-in for tanh / sigmoid), He / Kaiming (scaled
for ReLU). MLPL initializes weights via `randn(seed,
shape)` scaled by 0.5 inside `linear`; explicit
Xavier / He variants are deferred.

## Workspace

The full set of bindings and metadata in the current REPL
session: vars, params, models, tokenizers, optimizer state,
experiment log, frozen set, tag side-table, peer
dispatcher. `:wsid` shows the counts; `:vars`, `:models`,
`:tags`, `:experiments` list contents.
