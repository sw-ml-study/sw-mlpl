# Glossary

Short definitions for the ML terms that appear in MLPL demos
and lessons. Alphabetical. Each entry names the closest MLPL
construct so you can poke at the concept in the REPL. Concepts
that MLPL does not ship say so plainly and name the closest
shipped construct; concepts outside MLPL's
teaching-language scope (MLOps, deployment, monitoring) carry
an `out of scope` note.

## 3D Visualization Stage

A parallel 3D viewport showing computation history as
sculptural objects on a landscape stage. Toggle with `:3d`
(on) / `:2d` (off) or `Ctrl+3`. Each eval step places a
shape-proportional mesh: scalar = sphere, vector = bar,
matrix = rectangle, rank-3+ tensor = stacked slabs. Colors
encode the operation type. Arrow keys and nav buttons pan
the camera. Click a sculpture to see details (shape, rank,
elements, memory). The stage persists across tab switches;
`:clear` / Reset REPL clears it.

## :2d / :3d (REPL commands)

`:3d` opens the [[3D Visualization Stage]] (also Ctrl+3) and
`:2d` closes it. `:3d on` / `:3d off` are the explicit forms;
`:3d reset` re-centers the camera.

## argtop_k (builtin)

`argtop_k(scores, k)` returns the INDICES of the k largest
entries of a vector, descending -- the address half of a top-k:
where `top_k` keeps the winning values, `argtop_k` tells you
which positions won. Sampling shortlists, beam-style selection,
and retrieval hits are one `argtop_k` away.


## BDH (Dragon Hatchling)

A brain-inspired architecture built around SPARSE POSITIVE
activations plus a fast episode-local edge state updated by
Hebbian-style local rules -- separate from the trained weights.
The sparsity is the interpretability story: with few units
active, individual units and edges can be watched, ablated,
and causally tested. Not in MLPL.


## compress (builtin)

`compress(mask, a[, axis])` keeps the slices of `a` along
`axis` (default 0: rows) where the rank-1 `mask` is nonzero --
APL's compress. The selection half of [[Rejection Sampling
(best-of-N)]]: `compress(gt(scores, 0.9), C)` keeps exactly the
candidates a verifier accepted, and the companion targets follow
with the same mask.


## dedupe_rows (builtin)

`dedupe_rows(X)` -- the unique rows of a rank-2 `[n, L]` array,
first occurrence kept in original order, returned as a
`{rows, index}` record: `d.rows` is the deduplicated dataset,
and `gather_rows(Y, d.index)` carries any companion array
(targets, scores, difficulty) through the same selection. The
curation step between generation and [[Rejection Sampling
(best-of-N)]].


## emit_frame (builtin)

`emit_frame(name, step, x)` streams tensor `x` as one live
frame on a named animation channel when a server is connected
(the Game of Life demos stream their boards this way); with no
connection it is a no-op. It returns `x` unchanged, so it drops
into a pipeline without changing the math.


## Energy-Based Model (EBM)

Learn a scalar ENERGY (compatibility) function over
configurations, then treat low energy as "good": inference is
search/descent toward low-energy states rather than one
forward pass. The lineage runs [[Hopfield Network]] ->
Boltzmann machines -> modern latent predictors like [[JEPA]];
useful today as a ranking view -- score candidates by learned
compatibility instead of likelihood. Not in MLPL.


## engram_stats (builtin)

`engram_stats(e, ids)` (addressing + memory health) or
`engram_stats(e, h, ids)` (adds gate activity on hidden states
`h`) returns a record: `rows_addressed`, `unique_rows`,
`collisions`, `nonzero_rows`, `max_row_norm`, plus `gate_mean` /
`gate_max` in the three-argument form. The health panel for an
[[engram (builtin)]]: which memory rows the ids address, what
training wrote, and how far the gate opened.


## flatten (builtin)

`flatten(a)` -- ravel: every element of `a` as a rank-1 vector
in row-major order. The shape-erasing companion to reshape:
`reshape(flatten(a), dims)` is the general re-layout idiom. APL
heritage (monadic comma).


## floor / ceil / round (builtins)

Elementwise integer rounding: `floor(a)` rounds every element
down, `ceil(a)` up, `round(a)` to the nearest integer. All three
keep the input's shape.


## grade_up / grade_down (builtins)

`grade_up(v)` / `grade_down(v)` -- APL's grade: the STABLE
argsort index vector of a rank-1 `v`, ascending / descending
(ties keep original order). Grade returns WHERE things go rather
than moving them, so one grade orders any number of companion
arrays: `gather_rows(X, grade_up(difficulty))` is the
[[Curriculum Learning]] idiom, `gather_rows(C,
grade_down(scores))` ranks candidates best-first.


## Hopfield Network

The classic associative memory: patterns are stored as
attractors of an energy function, and recall runs the
dynamics from a noisy cue downhill to the nearest stored
pattern. Content-addressed and noise-tolerant -- the
complement of [[Engram]]'s exact hash addressing (modern
continuous Hopfield layers are attention-like). Not in MLPL.


## HRM (Hierarchical Reasoning Model)

Two coupled recurrent schedules -- a slow high-level planner
and a fast low-level worker iterating between plan updates --
so reasoning happens at two timescales. Conceptually a
composition of two [[TRM (Tiny Recursive Model)]]-style loops
rather than a new primitive. Not in MLPL.


## In-Context Reinforcement Learning (ICRL)

Improving an agent ACROSS attempts without touching its
weights: each try's outcome (reward, critique, failing tests)
is fed back into the next attempt's context, so the learning
loop lives in the prompt rather than the optimizer. The
reinforcement sibling of [[In-Context Learning (ICL)]]: ICL
selects good demonstrations; ICRL closes the loop with
evaluated feedback from the agent's own attempts. Not an MLPL
builtin; the pieces compose from experiments + evaluation.


## JEPA (Joint-Embedding Predictive Architecture)

Predict the REPRESENTATION of a future/masked observation
rather than the observation itself: a context encoder and a
target encoder meet in latent space, so the model learns
predictable structure without modeling every pixel. An
[[Energy-Based Model (EBM)]] descendant and a candidate
substrate for [[World Model]]s. Not in MLPL.


## kg_neighbors / kg_verify / kg_paths / kg_split (builtins)

The [[Knowledge Graph]] task oracle over plain `[E, 3]`
`(src, relation, dst)` edge arrays -- no graph value type needed.
`kg_paths(edges, hops, n, seed)` GENERATES `[n, hops+1]` valid
multi-hop paths (seeded random walks); `kg_verify(edges, paths)`
CHECKS candidate rows (1 where every consecutive pair is an
edge); `kg_neighbors(edges, node[, rel])` walks one hop; and
`kg_split(edges, frac, seed)` makes the generalization split --
`{seen, unseen}` with unseen edges touching entities the seen
side never contains, so held-out tasks cannot be answered by
memorization. Generate, verify, keep: pair with [[Rejection
Sampling (best-of-N)]] and `compress`.


## knn_graph (builtin)

`knn_graph(X, k)` builds an `[N*k, 3]` edge list of
`(i, j, dist)` rows -- each point's k nearest neighbors by
squared distance. The input layer for [[UMAP]]-style neighbor
embeddings, and the general starting point for any graph built
from raw points.


## Knowledge Graph

Entities and typed relations as a graph -- and, for small-model
training, an ORACLE: it can generate multi-hop reasoning
questions, verify answers by checking paths, grade difficulty
by hop count, and split train/eval by graph region to test
generalization instead of memorization. Not an MLPL builtin.


## loss_curve (builtin)

`loss_curve(losses)` renders a training-loss line chart from a
loss-per-step vector -- most often `loss_curve(last_losses)`
immediately after a `train` block, which records one loss per
step. See [[Loss]].


## mHC (manifold-constrained Hyper-Connections)

Hyper-Connections widen a transformer's [[Residual]] stream
into several parallel streams with learned mixing between
them; the manifold-constrained variant projects the mixing
matrices onto a stable set (Sinkhorn-style normalization
toward doubly-stochastic), trading raw freedom for training
stability at depth. DeepSeek lineage, like [[Engram]]. Not in
MLPL.


## Multi-Token Prediction (MTP)

Training a language model to predict SEVERAL future tokens per
position (extra heads for t+2, t+3, ...) instead of only the
next one. A denser training signal, and at inference the far
heads can PROPOSE a draft that the model verifies in one
parallel pass -- self-speculation, the same accept-or-reject
shape as [[Speculative Decoding]] without a separate draft
model. Not in MLPL.


## Pareto Frontier (efficient frontier)

The set of models no other model beats on EVERY axis at once
-- quality vs latency vs memory vs size. The right mental
model for constrained hardware: the question is never "the
best model" but "the best model under this machine's
constraints", and a plot of the frontier answers it at a
glance. MLPL: `pareto_front(P, dirs)` computes the frontier
mask over a metric matrix.

## param_count (builtin)

`param_count(m)` -> the total number of trainable parameters
across a model's `param` arrays, as a scalar. The size axis of
a quality-vs-parameters [[Pareto Frontier (efficient
frontier)]]: record it as a `*_metric` inside an `experiment`
block and pull it back with [[experiment_metric (builtin)]].

## experiment_metric (builtin)

`experiment_metric("name")` -> one recorded metric across the
in-memory experiment log, as a `[runs]` vector in run order.
Runs that did not record the metric are skipped; a metric no
run recorded yields the empty `[0]` vector. The bridge from
`experiment` blocks to arrays: column-concat several calls into
the `[n, k]` matrix that [[pareto_front (builtin)]] consumes.

## pareto_plot (builtin)

`pareto_plot(P, dirs)` renders the frontier picture: every row
of the `[n, 2]` metric matrix as a dot -- frontier members
highlighted and enlarged -- with the classic staircase line
stepped through the frontier, sorted by the first column. The
mask comes from [[pareto_front (builtin)]] internally, so the
plot and the mask can never disagree.

## pareto_front (builtin)

`pareto_front(P, dirs)` -> the `[n]` 0/1 mask of non-dominated
rows of the `[n, k]` metric matrix `P`. `dirs` gives one
direction per column: `1` = maximize (quality), `-1` = minimize
(parameters, loss). A row is dominated when some other row is
at least as good on every column and strictly better on one;
duplicate rows dominate neither way, so both stay. Composes
with the selection substrate: `compress(mask, P)` keeps the
frontier rows, `scatter_labeled(P, mask)` renders the frontier
highlighted. See [[Pareto Frontier (efficient frontier)]].

## pi / e (builtins)

`pi()` is 3.14159... and `e()` is Euler's 2.71828..., each a
zero-argument builtin (constants are functions in MLPL -- there
is no bare constant namespace). `sinusoidal_encoding` and the
math functions consume them like any other scalar.


## rand_ints (builtin)

`rand_ints(n, lo, hi, seed)` -- `[n]` uniform integers in
`[lo, hi)`, deterministic per seed (explicit PRNG state, so the
same seed gives the same bits on every platform). The integer
source for [[Synthetic Data]] generators: token ids, template
slots, corruption positions. `randn` is its Gaussian float
sibling.


## range (builtin)

`range(n)` -- the integers 0, 1, ..., n-1 as a vector: the
index generator most synthetic examples start from
(`reshape(range(12), [3, 4])` is the canonical toy tensor).
`iota` is a DEPRECATED alias from APL heritage; prefer `range`.


## Rejection Sampling (best-of-N)

Generate N candidates, score each with a verifier (exact
oracle, tests, reward model), then keep only what passes --
used three ways: pick the best answer at inference (best-of-N),
build clean training sets from noisy generators, and keep only
high-reward agent trajectories. The verifier's quality IS the
method's quality: a cheap deterministic check (does it compile?
does the graph path exist?) beats a vague score. Not an MLPL
builtin; compose from generation + evaluation.


## running_product (builtin)

`running_product(v)` -- the running (cumulative) product along
a rank-1 vector: `out[i]` is the product of `v[0..=i]`, same
length as the input. The multiplicative scan: a diffusion noise
schedule's alpha-bar is `running_product(alphas)`, and the
"Thinking in Arrays" demo uses it to absorb an associative time
loop into one expression. `cumprod` is the deprecated alias;
the additive sibling is `running_sum`.
See [[scan (higher-order)]].

## running_sum (builtin)

`running_sum(v)` -- the running sum along a rank-1 vector:
`out[i]` is the sum of `v[0..=i]`, same length as the input. The
additive scan: prefix sums, cumulative totals, and CDFs in one
call (`running_sum(prices * qty)` is revenue accumulated order
by order). Pairs with [[running_product (builtin)]]; `scan(:op)`
is the general form both specialize.


## Scaffolded Reasoning

Give the learner privileged structure DURING TRAINING --
worked steps, graph paths, execution traces, tool-call
transcripts -- then remove the scaffold at inference and test
whether the procedure was internalized. Distinct from
[[In-Context Learning (ICL)]] (scaffolds appear in training
data, not the prompt) and from [[Distillation]] (the teacher is
a structure, not a model). Not an MLPL builtin.


## Script Editor (web UI)

A multi-line text editor tab in the web playground for
writing, loading, and running `.mlpl` scripts. Buttons:
Run (executes all non-comment lines), Load (file picker
for `.mlpl` / `.txt`), Save (downloads as `session.mlpl`),
Clear. Comments start with `#`. The shebang line
`#!/usr/bin/env mlpl-repl` is skipped automatically.
Content persists across tab switches.

## abs (builtin)

Elementwise absolute value: `abs(x)` returns `|x|` for each
element. Pure scalar map; preserves shape.

## Accuracy

The fraction of predictions that are exactly right:
`mean(eq(predict_batch(m, X), Y))`. The first metric to check
and the easiest to fool -- on imbalanced data a model that
always predicts the majority class scores high while learning
nothing, which is why [[Precision vs Recall]], [[F1 score /
threshold tuning]], and [[ROC / AUC]] exist. MLPL:
`confusion_matrix(pred, Y)` shows where the misses live.

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

[[Adam]] with decoupled weight decay. Plain `adam` mixes
weight decay into the gradient before the moment update,
which interacts oddly with the per-parameter learning-rate
scaling; AdamW applies the decay directly to the parameter
update step, keeping it independent from the moment
scaling. Empirically robust default for transformer
training. **Deferred** in MLPL: `adam(loss, params, lr,
b1, b2, eps)` ships today; an `adamw` variant with explicit
`weight_decay` is on the regularization-tour roadmap.

## Adversarial Training

A training paradigm where two networks compete: the
Generator tries to produce realistic data and the
Discriminator tries to distinguish real from fake. Each
network's loss depends on the other's output, creating a
minimax game. In MLPL, this is implemented as two
alternating `adam()` calls in a single `train` block -- one
updates D's weights while G is fixed, then one updates G's
weights while D is fixed. See also: GAN.

## Adversarial examples

Inputs crafted with small, often imperceptible perturbations
that flip a model's prediction. The classic image-recognition
example: change a few pixels and "panda" becomes "gibbon".
MLPL ships `perturb_params(model, family, sigma, seed)` for
weight-space perturbation ; input-space adversarial
attacks are not in MLPL.

## args / list_get (builtins)

`args()` returns a `StrList` of the trailing CLI args passed
to the script after the `--` separator:
`mlpl-repl -f script.mlpl -- foo bar` makes `args()` return
`["foo", "bar"]`. Empty list when run from the interactive
REPL or the web playground. The list itself is read-only;
extract one element at a time with `list_get(args(), i)`.

`list_get(xs, i)` indexes into a `StrList` and returns the
`i`-th string wrapped in a [[Result type]]: `Ok(string)` when
`i < len(xs)`; `Err("list_get: index N out of bounds (list
has M items)")` when out of range. The Result wrap is what
makes the canonical missing-arg fallback work in one line:
`name = unwrap_or(list_get(args(), 0), "default-name")`.

Together with [[to_number / to_int (builtins)]] this lets a
script accept numeric command-line args:

```mlpl
epochs = unwrap(to_int(unwrap(list_get(args(), 0))))
lr     = unwrap(to_number(unwrap(list_get(args(), 1))))
```

## :ask / :connect (REPL commands)

`:ask <question>` sends the question -- plus your recent REPL
activity as context -- through the connected `mlpl-serve` to its
Ollama model, so answers are about your actual session rather
than generic trivia. `:connect list` shows the server's
installed models with the current pick marked; `:connect set
<model>` selects one for the session. Both need a connected
server; with no explicit pick the server auto-uses a
median-size installed model.

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
of keys. The classic formula is [[Softmax]] over `Q @ K^T /
sqrt(d)`, times `V`. MLPL builds it into `attention(d_model,
heads, seed)`
and [[Causal attention]]; the manual three-line version
runs in the "Attention Pattern" demo. See also
[[Multi-head attention]] for the `heads > 1` case.

`apply(mdl, X)` and [[attention_weights (builtin)]] accept both
rank-2 `[seq, d_model]` input and rank-3 `[B, T, d_model]`
batched input for any `heads >= 1` (`d_model` must be
divisible by `heads`). For rank-3 input each batch entry is
processed independently and the per-batch outputs are
stacked back. The tape lowering uses the [[Stack (tape op)]]
primitive for both the per-head join (axis 1) and the per-
batch join (axis 0), avoiding the O(N^2) cost of a chained
binary [[concat (builtin)]].

## Attention map

The `[T, T]` matrix of attention weights between every pair
of positions in a sequence. Renders cleanly as a [[Heatmap]].
Returned by [[attention_weights (builtin)]] in MLPL.

## attention_weights (builtin)

`attention_weights(model, X)` walks `model` to its first
`attention` / `causal_attention` layer, transforms `X`
through any preceding layers in the outer chain, and
returns just the softmax weight matrix -- without the
`@ V` value-multiplication and `@ Wo` output-projection
that `apply(model, X)` would do after it. surface;
step 013 generalized the return shape to four
cases:

| Input | heads | Output shape |
|-------|-------|--------------|
| `[T, d_model]`    | 1 | `[T, T]` |
| `[T, d_model]`    | h | `[h, T, T]` |
| `[B, T, d_model]` | 1 | `[B, T, T]` |
| `[B, T, d_model]` | h | `[B, h, T, T]` |

Each `[T, T]` slab is row-stochastic (softmax). Used by
the ViT attention-pattern demos to render heatmaps over
patch positions -- with multi-head, the `[heads, T, T]`
output feeds directly into `svg(_, "heatmap_grid")` for a
per-head 2x2 grid. Pair with the [[attention_overlay (viz type)]]
to see attention painted over the actual image.

## attention_overlay (viz type)

`svg(image, "attention_overlay", attn)` paints a translucent
viridis-colored heatmap on top of an image. `image` is a
`[3, H, W]` channel-first RGB tensor (the same layout
[[load_images (builtin)]] and `load_preloaded("pets_tiny")`
produce). `attn` is either `[P]` (single head) or `[heads, P]`
(multi-head), where `P = (H / patch) * (W / patch)` must be
a perfect square so it can be laid out as a sqrt(P) x sqrt(P)
patch grid. For multi-head input the output is a
ceil(sqrt(heads))-column grid of overlaid tiles, one cell per
head, labeled by head index.

Typical usage after training a Vision [[Transformer]]: pick a
test image, run [[attention_weights (builtin)]] to get
`[heads, T, T]`, reduce over the query axis to get
`[heads, T]` per-patch mean incoming attention, and feed
that to the overlay. Bright yellow patches are "where this
head pays attention on this image"; dark purple patches are
ignored. Companion to [[heatmap_grid (viz type)]] -- the
heatmap_grid shows the full [T, T] matrix per head; the
overlay shows where on the IMAGE each head looks.

## Autoencoder

A network trained to reconstruct its input through a low-
dimensional bottleneck (latent) layer. Used for compression,
denoising, and unsupervised representation learning. See also
`VAE`. Not an MLPL builtin.

## Autograd

Reverse-mode automatic differentiation. The runtime records
a tape during the forward pass; calling `grad(loss, wrt)`
walks the tape backwards to compute the gradient with respect
to a tracked parameter.

## Autoregression

Generating a sequence one element at a time, where each new
element is predicted from all the previous ones -- including the
model's own earlier outputs, fed back in as input. A language
model is autoregressive: it predicts the next token from the
tokens so far, picks one, appends it, and repeats. In training
this is "next-token prediction" -- [[Cross entropy]] of the
predicted distribution against the true next token at every
position at once (teacher forcing); at generation time the loop
runs for real, feeding each chosen token back in.

Not a primitive: MLPL builds it from existing pieces, so no new
builtin or demo is required. A [[Causal attention]] model (so
position `t` never sees `t+1`) plus a generation loop, e.g.
`repeat N { logits = apply(m, seq); nxt = sample(top_k(last_row(logits), 20), 0.8, step); seq = concat(seq, nxt) }`.
MLPL: `causal_attention` + `apply` + `last_row` + `sample` /
`top_k` + `concat`. See the "Tiny LM Generate" demo and the
generation steps of the LoRA fine-tune literate pages; the
tic-tac-toe demos apply the same idea to board states, not text.

Distinct from the classical-statistics sense (an AR(p)
time-series model, where a value is a fixed linear function of
its last p values). MLPL's neural, [[Sampling]]-driven
next-token form is the "AI autoregression" the demos use.

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
pass, [[Autograd]], [[Chain rule]].

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
batch norm is not in MLPL.

## batch / batch_mask (builtins)

`batch(x, size)` slices a `[N, ...]` array into a `[B, size,
...]` array of batches; the trailing batch is zero-padded if
`size` does not divide `N`. `batch_mask(x, size)` returns the
matching `[B, size]` `0 / 1` mask so downstream ops can ignore
the padded positions.

## Bayes' theorem

`P(H | D) = P(D | H) * P(H) / P(D)`: the posterior probability
of a hypothesis given data equals the likelihood of the data
under the hypothesis, times the prior, normalized. The entire
Bayesian toolkit is this one identity applied repeatedly -- see
[[Prior / Posterior / Likelihood]] and [[Naive Bayes]] (which
adds a feature-independence assumption). In array terms it is
elementwise multiply-then-normalize: `post = pr * lik /
reduce_add(pr * lik)`.

## Beam search

A decoding strategy that keeps the top-`k` partial sequences
at each step instead of committing to one. MLPL's generation
demos use greedy / multinomial sampling via `sample` + `top_k`
rather than beam search; beam search is not in MLPL.

## BERT

A specific encoder-only transformer architecture trained with
masked-token prediction. Pre-2020-era foundation for many
classification/QA tasks. MLPL builds tiny LMs in the demo
suite but does not ship pretrained model weights; loading
external checkpoints is not supported.

## Bias

The constant additive term in `y = x @ W + b`. A trainable
1-D parameter, smaller than the weight matrix; auto-tagged
`Bias`.

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

## Bottleneck (autoencoder)

The narrow middle layer of an [[Autoencoder]] where the
input is compressed to its smallest representation. The
encoder maps input to the bottleneck; the decoder maps
back. The bottleneck dimension controls the tradeoff
between compression and reconstruction quality. MLPL
demo: the Autoencoder (simple) demo compresses 8
dimensions to a 3-element bottleneck vector.

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
"Decision Boundary" / "[[K-Means]]" / "Moons MLP" demos.

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

## :builtins (REPL command)

Lists the built-in functions by category. `:describe <name>`
prints one builtin's signature and doc line; `:help` shows the
same list plus a syntax summary.

## BuiltinRef (`:foo` syntax)

A first-class-ish reference to a builtin or operator. Written
as `:` immediately followed by an identifier or one of
`+ * / -`. Examples: `:add`, `:max`, `:+`, `:sigmoid`. Used
as the first arg to higher-order builtins like `reduce(:op,
x[, axis])`. Lives in a separate namespace from regular
variables, so `add = 42` does not shadow `:add`. Forward-
compatible with first-class functions: when `Value::Function`
lands, `:foo` lifts to a function value.

## :clear (REPL command)

Resets the session: variables, models, and 3D state are all
cleared. Demos run in the SAME session until you clear it, so
later demos (and `:ask`) can see earlier results.

## Colon call (`:name(args)`)

Any builtin can be called directly from the REPL prompt by
prefixing a colon: `:disp(M)` is exactly `disp(M)`. Three colon
forms exist: `:command` runs a REPL command (`:vars`, `:help`),
`:name(args)` calls the builtin `name`, and a bare `:name` with
no parentheses is a [[BuiltinRef (`:foo` syntax)]] -- a
reference value for higher-order builtins like `reduce(:add,
x)`. `:name arg` with a space and no parentheses is none of the
three; the REPL answers it with a hint listing these forms.

## Calibration

How well a model's reported confidence matches its actual
accuracy. A well-calibrated classifier that says "70% sure"
is right 70% of the time. Modern neural nets are typically
overconfident. [[Temperature]] scaling on logits is the standard
post-hoc fix. Not an MLPL builtin.

## Catastrophic Forgetting

When fine-tuning a model on new data erases what it learned
from prior data. Mitigations: rehearsal (mix old + new
batches), elastic weight consolidation, low-rank adapters
(LoRA preserves the base by freezing it). MLPL's
`freeze` + `lora` workflow is the simplest defense.

## Causal attention

[[Self-attention]] with a lower-triangular mask before softmax so
position `t` cannot peek at `t+1`. Required for autoregressive
language models. MLPL: `causal_attention(d_model, heads,
seed)`.

## Chain (Model DSL)

Sequential composition of layers. `chain(linear(2, 8, 0),
tanh_layer(), linear(8, 2, 1))` is a 2-layer MLP.

## Chain rule

The calculus identity `d(f(g(x)))/dx = f'(g(x)) * g'(x)`,
applied recursively to compose gradients across the layers
of a neural network. [[Backpropagation]] IS the chain rule
applied to the computation graph: at each node, multiply the
incoming gradient by the local Jacobian, then propagate
upstream. MLPL's `grad(loss, wrt)` does this automatically;
the "Tiny MLP" lesson shows the manual two-layer version
where the hidden-layer gradient is `dZ1 = (dZ2 W2^T) * (1 -
H * H)` -- the tanh derivative being the local Jacobian
factor. See also Backpropagation, [[Autograd]].

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
`cross_entropy(logits, y)`. See "Linear [[Softmax]] Classifier"
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
fresh disjoint set of param names. "Neural Thicket"
ensembling clones a base model into 16 disjoint variants
before perturbing each.

## CNN (Convolutional Neural Network)

A network built around convolution and pooling layers,
designed for grid-structured data (images). MLPL does not
ship a `conv2d` layer today -- transformer + MLP families are
the model surface; convolutional layers are not in MLPL.

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

The window-and-sum operation at the core of CNNs. Not an
MLPL builtin.

## Cosine similarity

A similarity measure between two vectors that is invariant
to magnitude: `cos(u, v) = dot(u, v) / (norm(u) * norm(v))`,
range `[-1, 1]`. Distinct from `Dot product` (which is
sensitive to magnitude). The standard scoring function for
embedding-based retrieval (RAG) and nearest-neighbor lookup
on learned representations. MLPL: build from `dot`,
`sqrt`, `reduce_add` -- no dedicated builtin.

## Completion popup (REPL)

Press `Ctrl+Space` in the web playground's REPL input to
complete the token at the cursor -- the IDE-standard binding
(VS Code, IntelliJ, Emacs). `Tab` is reserved for browser
focus traversal.

Behavior: unique match -> inline insertion at the cursor;
ambiguous match -> a row of chips appears below the input.
Empty prefix (cursor on whitespace) does nothing.

Keybindings when the popup is open:

- `ArrowDown` / `ArrowUp` -- navigate the highlighted chip
  (wraps around at both ends)
- `Enter` -- accept the highlighted candidate at the cursor
- `ArrowRight` -- accept the highlighted candidate, but
  ONLY when the cursor is already at the end of the input;
  otherwise passes through as a normal cursor move
- `Escape` -- dismiss the popup without inserting anything
- Click on any chip -- accept that candidate (same as Enter
  on the highlighted one)

When the popup is closed, `ArrowUp` / `ArrowDown` navigate
command history as usual.

Candidate sources, in order: REPL slash-commands
([[:vars (REPL command)]], [[:introspect (REPL command)]], ...),
MLPL keywords (`train`, `repeat`, `experiment`, `for`,
`in`, `param`), all runtime builtins (every name in
`mlpl_runtime::runtime_builtin_names()`). Live user-bound
variable / model names are not included.

## Comparison ops: `gt`, `lt`, `eq` (builtins)

Elementwise predicates returning `0.0` / `1.0`. `gt(a, b)`,
`lt(a, b)`, `eq(a, b)`. MLPL has no boolean type -- the
`0 / 1` floats double as masks (multiply to filter) and
counts (`reduce_add` to sum a "how many true" tally).

## Cross-attention

[[Attention]] where the queries come from one sequence and the
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
priors, dimensionality reduction (PCA, [[t-SNE]]), and learned
representations.

## :describe (REPL command)

`:describe <name>` prints a typed summary of a binding -- shape
+ tag + values preview for an array, layer tree for a model,
vocab + merge count for a tokenizer, signature for a builtin, or
the one-line brief for a REPL command (`:describe history`).
The name accepts the colon spelling too: `:describe :disp` and
`:describe disp` are the same command. Per-tag bodies add detail
(Probability rows show the verified-or-violated row-sum
invariant; [[Gradient]] shows `wrt`, etc.).

## device block (language keyword)

`device("target") { body }` pushes a device target onto a
stack so ops inside the body dispatch through that backend.
MLPL: `device("cpu") { ... }` (default), `device("mlx") {
... }` (Apple MLX, ), and -- with `--peer` registered
-- a remote service peer . Bindings created inside
the block carry the device tag forward; cross-device ops
strict-fault.

## Data Augmentation

Generating extra training samples by transforming existing
ones (flipping images, masking tokens, paraphrasing text)
without changing the label. Cheap regularization. Not an
MLPL builtin.

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
the standard fix is ensembling, e.g. [[Random Forest]].
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

## depth (builtin)

`depth(x)` returns the nesting level of a value (APL2 sense):
`0` for a scalar, `1` for any flat array (vector, matrix, or
higher-rank). MLPL's `DenseArray` is flat, so depth
is `0` for a scalar and `1` for any array.
Pairs with `shape`, `rank`, `size`, and `tally` as the
structural-introspection set; `disp(x)` shows all of them at
once. APL heritage. Do NOT confuse depth with rank: a rank-5
dense tensor still has depth 1. Rank counts axes; depth counts
levels of nesting. Depth only exceeds 1 for ragged / nested
arrays (cf. `RaggedTensor`, `NestedTensor`), so there is no
everyday dense-tensor analog until then.

## disp (builtin)

`disp(x)` returns an ASCII box diagram (a string) that makes
the rank, shape, and depth of `x` visible: rank <= 2 renders
as a framed grid, rank >= 3 as a labeled stack of
leading-axis slices, with a `rank R  shape [..]  depth D`
footer. MLPL's answer to APL's `]display`. Print it in the
REPL to see an array's structure at a glance. (Unlike MATLAB's
`disp`, which just prints values, MLPL's `disp` draws the box
frame around rank / shape / depth -- closer to NumPy `repr` or
PyTorch `print` but structure-first.)

## Discriminator

The half of a GAN that classifies inputs as real or fake.
Takes data (real or generated) and outputs a score between
0 (fake) and 1 (real) via sigmoid. In MLPL's GAN demo,
`D = chain(linear(2, 8, seed), relu_layer(), linear(8, 1,
seed2))` maps 2D points to a scalar real/fake score. The
Discriminator trains on both real data and the Generator's
fakes, getting better at telling them apart. See also: GAN,
Generator.

## Dimensionality reduction

Project an `[N, D]` matrix down to `[N, k]` for some `k < D`,
usually `k = 2` or `3` so the result fits on a screen. Two
broad families. **Linear** methods rotate the data along
axes of maximum variance ([[PCA (Principal Component Analysis)]]
is the canonical example; `pca(X, k)` in MLPL). Fast, exact,
preserves global variance directions, but throws away every
non-linear structure -- a curved manifold gets flattened with
crossings. **Non-linear** / manifold methods build a local
neighborhood graph and optimize a low-D layout that preserves
that graph. [[t-SNE]] (`tsne(X, perp, iters, seed)`) inflates
local neighborhoods at the cost of global distance.
[[UMAP]] (`umap(X, n_neighbors, min_dist, iters, seed)`)
preserves both local AND global structure via fuzzy simplicial
sets + cross-entropy with negative sampling -- the recommended
modern default for visualizing learned embeddings. Multi-
dimensional scaling (MDS), Isomap, and Laplacian eigenmaps are
adjacent methods covered in the dim-reduction milestone's
Phase 5. See also [[Manifold preservation]] and
[[Swiss roll]] (the canonical "PCA fails, manifold methods
win" test bed) and the "Dimensionality reduction" learning
path for a walk through the demos.

## DDPM (Denoising Diffusion Probabilistic Model)

The standard training/sampling recipe for [[Diffusion Models]]:
train a network to predict the noise added to a sample at a
random timestep (MSE of predicted vs actual noise), then sample
by starting from pure noise and applying the learned reverse
step down the [[Noise schedule]]. DDIM is a faster,
deterministic sampler over the same trained model. MLPL: the
"Diffusion (2D points)" demo trains an MLP noise-predictor and
reverse-samples on a CPU-runnable 2D dataset.

## Diffusion Models

Generative models that turn noise into data by reversing a
gradual noising process. The FORWARD (noising) process adds
Gaussian noise over a [[Noise schedule]] until the sample is
pure noise; a network then learns the REVERSE (denoising)
process, predicting and removing noise step by step to
synthesize new samples. The non-autoregressive counterpart to
next-token generation (see [[Autoregression]]) -- it refines a
whole sample at once rather than emitting it left to right.
State of the art for images/video. MLPL builds the algorithm
from existing pieces (a `linspace`/`running_product` schedule, an MLP
denoiser, `adam`); the in-browser "Diffusion (2D points)" demo
teaches it on tiny data; the same recipe scales
to image U-Nets and real text-to-image on a connected GPU.
See [[DDPM]].

## Distillation

Training a smaller "student" model to imitate a larger
"teacher" model's outputs (logits or probabilities) instead
of the original labels. Compresses knowledge into faster /
cheaper models. MLPL has no distillation pipeline builtins.

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
scope; preference-data builtins are not in MLPL.

## Dropout

A regularization technique that zeros out a random fraction
of activations during training so the network learns
distributed representations. Not an MLPL builtin.

## Early Stopping

Halting training when validation loss stops improving (rather
than after a fixed step count) to avoid overfitting. MLPL
doesn't ship a built-in early-stop hook; the user can
condition the `train` body on `last_losses` patterns.

## embed_table (builtin)

`embed_table(model)` walks a `ModelSpec` tree depth-first
left-to-right and returns the first `Embedding` layer's
`[vocab, d_model]` matrix as a plain array. .5
shipped this so demos can inspect / project / cluster a
learned embedding after training.

## Eigenvalues / Eigenvectors

Directions a matrix only stretches, never rotates:
`matmul(A, v) = lambda * v`. The eigenvectors of a dataset's
covariance matrix are its principal components ([[PCA
(Principal Component Analysis)]]), and the largest one can be
found by POWER ITERATION -- repeatedly multiply a random vector
by the matrix and normalize: `v = matmul(A, v) / sqrt(sum(v *
v))`. Not an MLPL builtin; the power-iteration idiom is a few
lines of `matmul`.

## Embedding

A learned `[vocab, d_model]` lookup table that maps token ids
to dense vectors. MLPL: `embed(vocab, d_model, seed)` is a
Model DSL layer; `embed_table(model)` returns the underlying
`[vocab, d_model]` matrix.

## env (builtin)

`env(name)` reads the OS environment variable `name` and
returns a [[Result type]]: `Ok(string-value)` if set,
`Err("env: NAME not set")` if unset. Lets a script read
configuration from the shell environment without baking it
into the source.

Canonical scripting pattern with a default fallback:

```mlpl
model_path = unwrap_or(env("MODEL_PATH"), "default-model.bin")
```

Pair with [[args (builtin)]] to give a
script both `--flag value` and `$ENV` configuration surfaces.

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
decoder. See "[[Decoder / encoder]]" for the role of each side.

## Ensembling

Running multiple trained models on the same input and
combining their outputs (averaging logits, voting). Often
beats any single member at the cost of inference time and
memory. The classical family: BAGGING trains members on
resampled data ([[Random Forest]]), BOOSTING trains them
sequentially on the last one's mistakes ([[Gradient
Boosting]]), STACKING learns a combiner model over their
outputs. MLPL's "Neural Thicket" demo runs a 16-member
weight-perturbed ensemble end-to-end.

## Error Handling

sw-MLPL splits errors into two planes. HARD errors (wrong arity, shape mismatch, out-of-bounds index) stop the line loudly. [[Result]] VALUES -- `ok(v)` / `err(e)` -- carry failure as ordinary data: `is_ok` branches, `unwrap_or` gives a total read with a fallback, and `get_value`/`get_error` project each side as a 0-or-1 element vector (absence is emptiness, the APL2 zilde flavor). Two bridges cross the planes: `try { body } catch e { handler }` demotes a hard error into the record `e = {kind, message}`, and postfix `?` propagates an Err out of the enclosing `u:` function (the railway pattern). The convention for user-defined functions: validate inputs FIRST and return `err({kind, ...})` -- the caller decides, not the callee. See the "Error Handling (two planes, two bridges)" demo and docs/error-handling.md.

## Epoch

One full pass through the training dataset. Distinct from
"step" -- a step is one optimizer update. Epochs are dataset-
relative; steps are gradient-relative.

## estimate_train / estimate_hypothetical / feasible / calibrate_device (builtins)

"feasibility" surface. `estimate_train(model, steps,
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

## Feature engineering

Constructing model inputs from raw data: scaling, encoding
categories ([[One-hot encoding]]), crossing columns, extracting
counts and ratios. Deep learning shifted much of this work into
learned layers, but at small scale good features still beat
extra parameters. MLPL: `concat` / `reshape` / comparison masks
build feature columns; the [[Kernel method]] entry shows feature
maps standing in for kernels.

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
the parameter-efficient form .

## Feature map

The output of a convolutional layer: one 2D grid per filter,
each highlighting a different pattern (edges, textures,
shapes) in the input. A conv2d layer with 16 filters
produces 16 feature maps. Stacking conv layers builds a
hierarchy: early maps detect edges, later maps detect
complex structures. MLPL: the `[B, C_out, H, W]` output of
`conv2d` is a batch of feature maps.

## Flash Attention

A re-implementation of attention that fuses the softmax with
the matmuls and tiles to keep working memory in fast SRAM,
reducing both wall-clock and memory cost without changing
the math. MLPL's MLX backend uses naive attention;
fused / flash variants are not implemented.

## fill / zeros / ones (builtins)

Constant-array constructors. `zeros([d0, d1, ...])` makes a
zero-filled tensor of the given shape; `ones(...)` is one-
filled; `fill(shape, value)` is the general form. Used to
allocate accumulators (`losses = zeros([16])`), bias inits,
mask scaffolding.

## :fns / :list (REPL commands)

`:fns` lists your `def u:` functions with signatures and
doc-strings (APL's `)FNS`); `:list u:name` prints one function
back verbatim, `#` comments included.

## :help (REPL command)

`:help` prints the REPL command list and a syntax summary;
`:help <topic>` gives focused help (vars, models, fns,
builtins, describe, wsid); `:<cmd> --help` prints one command's
usage line.

## :history (REPL command)

Lists the recent REPL command lines. The same listing is handed
to `:ask` as session context.

## for / in (language keyword)

`for row in dataset { body }` streams over rows (or batches)
of a dataset, binding the row to `row` per iteration. The
last value of `body` is captured into `last_rows` for
plotting / inspection. added this construct; the
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
`unfreeze(model)` from .

## Game of Life

Conway's cellular automaton and the array-language world's favorite party trick: the APL2 one-liner computes every cell's next generation simultaneously. In MLPL the same shape falls out of [[rotate (builtin)]]: shift the whole board 8 ways, sum the shifted boards into a neighbor-count matrix N, and the entire rule is `gt(eq(N, 3) + G * eq(N, 2), 0)` -- birth on 3 neighbors, survival on 2 or 3. The "Game of Life (APL classic)" demo builds it three ways and animates 24 generations with `svg(F, "life")`. The rule, the neighbor count, and the animation are all ordinary array values.

## GAN (Generative Adversarial Network)

A framework where two networks train simultaneously in
competition. The Generator creates fake data from random
noise; the Discriminator classifies inputs as real or fake.
The Generator improves by fooling the Discriminator; the
Discriminator improves by catching fakes. At equilibrium
the Generator produces data indistinguishable from real.
MLPL: the "GAN (2D circle)" demo trains both networks with
alternating `adam()` calls inside one `train` block. See
also: Generator, Discriminator, Adversarial Training.

## Gaussian / Bernoulli / Uniform (distributions)

The three distributions most ML math leans on. GAUSSIAN
(normal): the bell curve, closed under sums, the default noise
model -- `randn(seed, shape)` samples it. BERNOULLI: a single
0/1 outcome with probability p -- `lt(randn-free uniform, p)`
masks sample it. UNIFORM: every value in a range equally likely
-- `rand_ints(n, lo, hi, seed)` is the integer version.
Histograms make each visible: `hist(randn(0, [1000]), 20)`.

## Generator

The half of a GAN that creates fake data. Maps random noise
(from the latent space) through learned weights to produce
output that should resemble real data. In MLPL's GAN demo,
`G = chain(linear(2, 8, seed), relu_layer(), linear(8, 2,
seed2))` maps 2D noise to 2D points approximating a circle.
The Generator never sees real data directly -- it only
receives gradient signal through the Discriminator.

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
GQA / MQA are not in MLPL.

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

## Gallery (viz output)

`svg(images, "gallery")`  renders an
`[N, 3, H, W]` image batch as an SVG grid of RGB
thumbnails. Each batch entry is laid out in a ceil-sqrt
grid, downsampled via block averaging so a 20-image
pets_tiny slice doesn't emit tens of thousands of unique
`<rect>` elements. Pixel values are interpreted in the
`[-1, 1]` normalized space that `load_preloaded("pets_tiny")`
and `load_images` produce; out-of-range values clamp instead
of wrap.

3-arg form `svg(images, "gallery", overlay)` (step
011) attaches a small text caption under each thumbnail.
`overlay` is `[N]` (one integer per image) or `[N, K]` for
K up to 4 (e.g. `[N, 2]` of actual / predicted). Values
render as integers separated by `/`; class-name mapping is
left to the caller -- pass a normalization-free integer
tensor and document the mapping in your demo's takeaway.

## Gradient Clipping

Capping the L2 norm of the gradient vector before applying
the optimizer update. Prevents explosive updates from rare
huge-gradient batches. Not an MLPL builtin.

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

## heatmap_grid (viz type)

`svg(data, "heatmap_grid")`  renders a
rank-3 `[N, R, C]` tensor as a grid of N heatmaps. Grid
layout is `cols = ceil(sqrt(N))`, `rows = ceil(N / cols)`
(so 2x2 for N=4, 3x3 for N=9, etc.). Each cell carries its
own min/max colormap so a sharply-focused panel and a
diffuse panel both render with visible structure rather
than washing one out. Each cell has a `head <i>` label
above it.

Driving use case: multi-head `attention_weights(model, X)`
returns `[heads, T, T]`; passing that to heatmap_grid lays
out one heatmap per head so per-head specialization is
legible at a glance. Used by the
`vit_attention_pattern_multihead.mlpl` (untrained baseline)
and `vit_multihead_quick.mlpl` (post-training) demos.

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

## If expression

`if cond { then } else { else }` is MLPL's branching primitive.
Like Rust, it is an EXPRESSION not a
statement -- it returns the value of whichever branch was
taken, so it composes into bindings: `x = if flag { 100 }
else { 200 }`. The `else` clause is REQUIRED; a dangling-if
would force a unit-typed branch which MLPL does not have.

`cond` is truthy when:

- A scalar `Number` that is non-zero (negatives count as
  truthy; only `0.0` is falsy).
- A [[Result type]] in its `Ok` state. `Err(_)` is falsy.

All other types (vectors, matrices, strings, records, etc.)
raise a runtime error. Use `is_ok(r)` to explicitly convert a
Result to a 0/1 scalar if you want to combine it with arithmetic.

Both branches are body sequences (the final statement's value
is the branch value), same convention as `repeat` / `train` /
`for` blocks. Either branch can return any `Value` type --
strings, vectors, Records, Results -- not just scalars.

## Synthetic Data

Training data GENERATED rather than collected: templates,
grammars, graph-derived questions, teacher-model output --
validated by an oracle before training (see [[Rejection
Sampling (best-of-N)]]). The small-model lesson: data quality,
structure, and ordering ([[Curriculum Learning]]) can buy back
much of what parameter count gives up. MLPL's synthetic
builtins (`moons`, `blobs`, `circles`, `grid`) are the
tiny-scale version of the idea.


## Test-Time Compute

Spending more computation per query at INFERENCE time to buy
accuracy -- longer reasoning chains, best-of-N sampling with a
verifier, iterative refinement loops, recursive models that
"think longer" on hard inputs. The dual of scaling parameters:
a small model with a good test-time strategy can outperform a
larger one-shot model on reasoning tasks. See [[Rejection
Sampling (best-of-N)]].


## train_val_curve (builtin)

`train_val_curve(train, val)` renders the training and
validation loss curves on one chart; the gap between the two
lines is the overfitting picture (see the "Watch a Model Learn"
demos). Pair with `val_split` to hold out the validation set.


## TRM (Tiny Recursive Model)

A small network applied REPEATEDLY to its own latent state --
parameter sharing across steps -- so depth becomes iteration
and hard inputs can get more passes. The minimal form of
recursion-as-reasoning: fixed or learned halting, optional
per-step losses. A [[Test-Time Compute]] strategy in
architecture form. Not in MLPL.


## While loop

`while cond { body }` re-evaluates `body` until `cond` is
falsy. Truthiness matches [[If expression]]
-- scalar non-zero or `Ok(_)`. The loop expression evaluates
to scalar `0` on normal exit, or to the `break value` if the
body exited via `break value`.

`break` and `continue` are the loop-control keywords:

- `break` exits the nearest enclosing `while` immediately.
  Bare `break` yields `0`; `break value` yields the supplied
  value (any [[Value type]]).
- `continue` skips the rest of the current iteration; the
  condition is re-checked from the top.

Using `break` or `continue` outside a `while` is a runtime
error (`break used outside of a while loop`). Loops do NOT
introduce a new variable scope -- assignments in the body
persist into the surrounding environment, matching the
`repeat` / `train` / `for` convention.

This is MLPL's general looping primitive; combine with
[[If expression]] for conditional break-out, [[Result type]]
for fallible inputs, and [[args() builtin]] for CLI-driven
iteration.

## Script exit codes

In `mlpl-repl -f script.mlpl` mode, the
process exit code is determined by the script's final value:
`Err(msg)` exits `1` and writes `msg` to stderr; everything
else exits `0`. The `exit(code)` builtin short-circuits with
the caller-chosen code (must be `0..=255`).

This makes MLPL scripts compose with Unix tooling:
`mlpl-repl -f check.mlpl && echo ok`, `... || echo "exit $?"`,
`echo input | mlpl-repl -f filter.mlpl`. See also
[[Stdin reading]], [[Result type]], and the `print` / `eprint`
builtins for the input/output side.

## Stdin reading

The `read_stdin()` and `read_stdin_lines()` builtins consume the script's stdin to EOF. `read_stdin()`
returns the whole input as a [[Value::Str]]; `read_stdin_lines()`
splits on `\n` and returns a [[Value::StrList]] with a trailing
empty entry stripped. Both refuse to read from an interactive
TTY -- they return `Err("...stdin is a terminal; pipe input or
use args() instead")` so the REPL doesn't hang on a stray
`read_stdin()`.

The intended shape is shell pipes: `cat data.txt | mlpl-repl -f
filter.mlpl`. Combine with [[Script exit codes]] for
`set -e`-style chained scripts.

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
SERVING is inference behind a network endpoint: `mlpl-serve`
does exactly this for connect-mode sessions (the browser sends
programs, the server evaluates and streams results back).

## :introspect (REPL command)

`:introspect` is a bundle command that
concatenates the output of every no-arg inspector into one
markdown-headered dump. Sections in fixed order:
[[:version]], [[:wsid (REPL command)]], `:builtins`,
[[:vars (REPL command)]], [[:models / :tokenizers (REPL commands)]],
[[:experiments (REPL command)]], [[:tags / :untag (REPL commands)]].
Each section prints under a `## :<topic>` header so a long
scroll stays scannable. Arg-taking inspectors
([[:describe (REPL command)]], `:untag`, `:help`) are NOT
included -- they need user input and don't fit the "dump
everything" intent. Useful as the optional last line of any
demo: when a notebook captures full workspace state you only
need one command.

## Interpretability / Mechanistic Interpretability

Mechanistic interpretability is the program of reverse-
engineering trained neural networks circuit by circuit:
identifying attention heads, feature directions, and
algorithms the model implements. The "Embedding exploration"
demo is a tiny taste -- [[t-SNE]] / k-NN over a learned
embedding table. Full circuit-level work is out of MLPL's
scope.

## Jacobian / Hessian

The gradient generalized. The JACOBIAN of a vector-valued
function is the matrix of every output's partial derivative
with respect to every input -- backpropagation is repeated
Jacobian-transpose-times-vector products, which is why reverse
mode never materializes the full matrix. The HESSIAN is the
matrix of second derivatives of a scalar loss; its eigenvalues
describe curvature (sharp vs flat minima). MLPL's autograd
computes gradients (first order); full Jacobians and Hessians
are not built in.

## Johnson-Lindenstrauss Lemma

A geometric guarantee: for any `N` points in high-D space,
a random projection to `k = O(log N / eps^2)` dimensions
preserves all pairwise distances within a `1 +- eps` factor
with high probability. The construction is trivial -- a
Gaussian random matrix scaled by `1/sqrt(k)` -- but the
guarantee is sharp. MLPL: `random_projection(X, k, seed)`.
The practical payoff: random projection is the right
SANITY BASELINE for any learned dim-reduction method. If
your fancy autoencoder / PCA / UMAP does not beat random
projection on the downstream task, your method is not
adding signal beyond raw geometric compression. See also
[[Multidimensional Scaling]] and [[Dimensionality reduction]].

## Jailbreaks

Prompt patterns that trick an LLM out of its safety training
("ignore previous instructions", role-play attacks). LLM-
safety territory; MLPL exposes `llm_call` but doesn't ship
a jailbreak / safety-eval surface.

## iota (builtin)

`iota(n)` returns the integer sequence `[0, 1, ..., n-1]` as
a rank-1 vector. The most basic array constructor; building
block for indexing / shape arithmetic / one-hot scaffolding.

## Kernel method

Computing inner products in a high-dimensional feature space
without ever visiting it: `K(a, b) = <phi(a), phi(b)>` for some
feature map phi. The trick behind nonlinear [[SVM (Support
Vector Machine)]]s. At whiteboard scale the feature map can be
EXPLICIT instead -- append squared and product columns
(`concat(X, X * X, 1)`) and a linear model in the widened space
is a nonlinear model in the original one. That explicit-phi
idiom is how MLPL demos express it.

## Key (K)

One of the three projections in attention. Each token emits
a key advertising "what I have to offer"; the dot product `Q
@ K^T` measures how strongly each query matches each key,
producing the unnormalized score matrix. In MLPL:
`K = matmul(X, Wk)` where `Wk` is `[d_model, d_model]` for
single-head or `[d_model, d_k]` per head. Paired with Query
(Q) and [[Value (V)]].

## K-Means

Unsupervised clustering by alternating "assign each point to
its nearest centroid" and "move each centroid to the mean of
its assigned points". The K-Means demo runs ten iterations.

## knn (builtin)

`knn(X, k)` returns each row's `k` nearest non-self neighbors
sorted by ascending distance with lower-index tie-break.
`[N, k]` integer-index output. ships this for
embedding inspection.

## KV Cache

The cache of past Key and Value tensors a transformer keeps
during autoregressive decoding so each new token only needs
to compute its own row, not the entire `[T, T]` attention
matrix. MLPL's "Tiny LM Generate" demo recomputes from
scratch each step; MLPL has no KV cache.

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
regularizes. Not an MLPL builtin.

## Labels

Ground-truth integer class indices for a classification task.
A `Labels { num_classes }` tag carries the class
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
LayerNorm proper is not in MLPL.

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
text as a `Value::Str` string. CLI-only: the browser
playground cannot fetch cross-origin URLs.

## load / load_preloaded (builtins)

`load("rel.csv")` / `load("rel.txt")` reads through an
`Environment::data_dir` sandbox set by the terminal REPL's
`--data-dir` flag. `load_preloaded("name")` serves
compiled-in corpora for the web REPL where filesystem access
is unavailable. Both produce a string for `.txt` and a
DenseArray (with header autoparse) for `.csv`.

`load_preloaded("pets_tiny")`  returns a
`Value::Record` with three fields: `X` (a `DenseArray` of
shape `[200, 3, 64, 64]` with `[batch, channel, y, x]` axis
labels), `Y` (a `[200]` label vector; `0 = cat`, `1 = dog`),
and `names` (a `Value::StrList` of source filenames). The
fixture is shipped as pre-decoded u8 RGB bytes via
`include_bytes!` so the WASM REPL has it without any live
decoder.

## fetch_dataset (builtin)

`fetch_dataset(name)` (step 004, native-only via the
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
`[B, N, P*P*C]` patch tokens for a [[ViT (Vision Transformer)]].
`P` is the square patch side length; it must divide both `H`
and `W`,
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

`concat(a, b)` joins two rank-0 or rank-1 arrays into a 1-D
vector; used by generation loops to append a sampled token id
to the growing sequence.

`concat(a, b, axis)` is the axis-aware extension. Both inputs
must agree on every dim except `axis`,
where the sizes add. Initial release supports `axis` in
`{0, 1}` only; higher axes are a follow-up. Differentiable on
the tape: forward stacks data per the axis layout; backward
splits the upstream gradient at the seam (`left_size` along
`axis`) and delivers each half to its parent. The driving use
case is CLS-token prepending in ViT: `concat(cls, patches, 1)`
adds a learnable `[B, 1, D]` token to the front of a
`[B, N, D]` patch sequence so the classifier head can read off
the CLS row after attention.

## predict_batch (builtin)

`predict_batch(model, X)`  runs a forward
pass through `model` and returns argmax over the trailing
axis as integer class indices. Equivalent to
`argmax(apply(model, X), last_axis)` but a single builtin
call so demos read cleanly. Not differentiable -- use
`apply(model, X)` inside `grad()` or `adam()` instead.
Driving use case: pair with `eq(preds, Y)` + `reduce_add`
to compute classification accuracy, or pass to the 3-arg
`svg(X, "gallery", overlay)` form for a labeled gallery.

## print / eprint (builtins)

`print(v)` and `eprint(v)` write `v`'s display form to stdout
and stderr respectively, followed by a newline. Both return
`v` unchanged so they compose into expressions:
`x = print(some_computation)` both binds `x` and shows the
value, without needing a separate sequencing block. Same
display contract as the REPL prompt -- a vector prints as
`1 2 3`, a matrix as space-and-newline-delimited rows, a
`[[Result type]]` as `Ok(...)` / `Err(...)`.

Driving use case: `mlpl-repl -f script.mlpl` only displays
the script's FINAL expression by default; `print()` is how a
script surfaces intermediate values to its operator. Pair
with `eprint()` to keep diagnostic messages out of the
script's main output stream so `mlpl-repl -f s.mlpl | grep
...` works cleanly.

## load_images (builtin)

`load_images(dir, [H, W])` (step 003, native-only via
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
the softmax. A `Logit` tag -- `cross_entropy`,
`sample`, and `top_k` expect them; passing `softmax`
output instead is the canonical double-softmax bug.

## LogProbability

The log of a probability. Numerically stabler than
multiplying probabilities; `log_softmax` produces them.

## LoRA (Low-Rank Adaptation)

Parameter-efficient fine-tuning. Replace each `Linear` with
`y = x @ W + (alpha/rank) * x @ A @ B + b` where `A`, `B`
are trainable low-rank adapters and `W`, `b` are frozen.
MLPL: `lora(model, rank, alpha, seed)`.

## Loss

A scalar that summarizes how wrong a model's predictions
are. Minimization target for `adam` / `momentum_sgd`.
Auto-tagged `Loss(kind)`.

## Loss Landscape

The high-dimensional surface defined by parameter values
mapped to loss values. Modern theories describe it as full
of saddle points, narrow valleys, and wide flat minima --
the latter generalize better. Tools to probe the landscape
(loss surface sharpness, Hessian eigenvalues, SAM) are
research targets.
MLPL renders it with `loss_landscape(surface, dims, path)`:
a 2-weight loss surface as a heatmap with the optimizer's
trajectory drawn on top.

## Lottery Ticket Hypothesis

Empirical claim that inside a randomly-initialized dense
network there exist sparse subnetworks ("winning tickets")
that, trained from scratch with their original initialization,
match the full network's accuracy. Drives interest in
pruning + retraining.

## Manifold Hypothesis

Real-world high-dimensional data (images, text, audio) lies
near a much-lower-dimensional manifold inside the ambient
space. Justifies dimensionality reduction (PCA, [[t-SNE]], UMAP)
and explains why deep learning works at all -- the network
need only be expressive on the manifold, not the cube. A
HOMEOMORPHISM is a continuous, invertible deformation
(stretching without tearing); layers that are homeomorphisms
can untangle a manifold but never change its topology -- which
is one lens on why width and nonlinearity matter.

## Manifold preservation

A property of a dimensionality-reduction method: distances and
neighborhoods on the underlying manifold are preserved in the
low-D projection. Linear methods (PCA) preserve the GLOBAL
ambient axes of variance but slice through curved manifolds,
smearing them in projection. Non-linear / manifold methods
([[t-SNE]], [[UMAP]], Isomap, Laplacian eigenmaps) build a
local neighborhood graph FIRST -- distances on that graph
approximate distances on the manifold -- then optimize a low-D
layout that preserves the graph. UMAP additionally preserves
GLOBAL inter-cluster distance via its repulsive (negative-
sampling) term, which is what the "UMAP vs t-SNE" demo
illustrates. The [[Swiss roll]] is the canonical test bed for
the linear-vs-manifold contrast.

## map (higher-order)

`map(:op, x)` -- elementwise apply of a unary BuiltinRef
across every element. Not an MLPL builtin; compose with
`reduce(:add, x * x)` or named math primitives (`exp(x)`,
`sigmoid(x)`) instead.

## Mask

A `0 / 1` (or `0.0 / -inf`) array that nullifies positions in
a downstream op. [[Causal attention]] applies a lower-triangular
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
MLPL stores everything in f64; mixed precision is
not supported.

## mean (builtin)

`mean(x)` returns the arithmetic mean of all elements as a
scalar. Distinct from `reduce_add(x) / shape(x)` only in
that it ignores axis arguments today (always full reduction).

## MLE vs MAP

Two ways to pick parameters from data. MAXIMUM LIKELIHOOD
(MLE) picks the parameters under which the observed data is
most probable -- minimizing [[Cross-entropy]] IS maximum
likelihood for classification. MAXIMUM A POSTERIORI (MAP)
multiplies in a [[Prior / Posterior / Likelihood|prior]] first;
an L2 penalty on weights is exactly MAP with a Gaussian prior,
which is why [[Weight Decay]] has a Bayesian reading.

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

[[Gradient descent]] with a running velocity. `momentum_sgd(loss,
params, lr, beta)` accumulates a momentum vector that
smooths out gradient noise.

## MSE (Mean Squared Error)

Regression loss: `mean((pred - target)^2)`. Used when the
target is continuous rather than a discrete class.

## Multidimensional Scaling

A classical dim-reduction method (Torgerson 1952, Kruskal
1964): given an `[N, D]` matrix, find `[N, k]` coordinates
that preserve all pairwise distances as faithfully as
possible. The classical (metric) MDS solves this via
eigendecomposition of the double-centered squared-distance
matrix; the SGD variant minimizes the stress
`S = sum_{i<j} (||Y_i - Y_j|| - d_ij)^2` directly. MLPL ships
the SGD variant: `mds(X, k, iters, seed)` returns
`[N, k]`. MDS preserves PAIRWISE DISTANCES rather than
variance directions ([[PCA (Principal Component Analysis)]])
or local neighborhoods ([[t-SNE]] / [[UMAP]]). When the
question is "which points are far from which?" -- e.g.,
psychophysics similarity judgments -- MDS is the right tool.
Related: [[Johnson-Lindenstrauss Lemma]] (random projection
is a degenerate case where distances are preserved
probabilistically rather than via optimization).

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

## Noise schedule

The sequence of variances added at each step of a [[Diffusion
Models]] forward process. A linear schedule picks betas with
`linspace(beta_min, beta_max, T)`; the alphas are `1 - betas`
and the cumulative product `alpha_bar = running_product(alphas)` gives
the one-shot noising `x_t = sqrt(alpha_bar_t) * x_0 +
sqrt(1 - alpha_bar_t) * noise`. Cosine schedules add noise more
gently early on. MLPL: `linspace` + `running_product` build it directly;
see the "Diffusion (2D points)" demo.

## Object detection (IoU)

Finding WHERE things are, not just what: predict a box (and
class) per object. Scored by INTERSECTION OVER UNION -- overlap
area divided by union area of predicted and true boxes; a
detection counts when IoU clears a threshold. Anchor boxes,
non-max suppression, and the YOLO/R-CNN families build on this.
Not an MLPL capability; [[Segmentation]] is the per-pixel
sibling.

## One-hot encoding

Converting an integer class index to a vector with 1.0 at
that index and 0.0 elsewhere. MLPL: `one_hot(labels,
num_classes)`.

## Oxford-IIIT Pet dataset

7,393 photographs of cats and dogs (12 cat breeds + 25 dog
breeds, ~200 images per breed), released by the Visual
Geometry Group at Oxford. Standard cat-vs-dog classification
benchmark with breed-level subclasses. MLPL uses it as the
training set for the Vision [[Transformer]] demos. The
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
area; MLPL has no detection builtins.

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
shape-and-position heuristics in .

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
top-k projection `[N, k]`. Companion builtins:
`pca_components(X, k)` returns the `[k, D]` LOADINGS matrix
(row `i` = direction of the i-th principal component in
original feature space), and `pca_variance_explained(X, k)`
returns the per-component fraction-of-variance vector
`[k]`. The "PCA via Power Iteration" demo writes the
projection out by hand; the loadings + variance helpers feed
the critical-dimensions heatmap.

## perturb_params (builtin)

`perturb_params(m, family, sigma, seed)` walks `m`'s param
tree, filters by `family` (`"all_layers"`, `"attention_only"`,
`"mlp_only"`, `"embed_and_head"`), and adds `sigma * randn(seed
+ i, shape)` to each matching param in place. Used by the
weight-perturbation ensembling pattern.

## Perceptron

A single linear layer plus a step / sigmoid activation -- the
1958 ancestor of every neural network. Limited to linearly-
separable problems alone; the famous Minsky-Papert XOR
critique drove research into multi-layer networks (MLPs).

## Perplexity

The exponentiated cross-entropy of a language model on a
held-out corpus: `exp(cross_entropy_loss)`. Standard LM
evaluation metric. Lower is better. MLPL ships
`perplexity(logits, targets)` as a convenience -- it returns
`exp(cross_entropy(logits, targets))` in one call.

## Pooling

Controlled information loss: shrink spatial dimensions by
taking the max or average over small non-overlapping windows.
Max pooling keeps the strongest activation in each window;
average pooling keeps the mean. Reduces computation and adds
translation invariance. MLPL: `pool2d(input, size, mode)`
where `mode=1` is max and `mode=0` is average.

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
harmonic mean. [[ROC / AUC]] summarizes the precision-recall
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
[[Fine-tuning]]: continue training on a smaller, task-specific
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

## Prior / Posterior / Likelihood

The three moving parts of [[Bayes' theorem]]. The PRIOR is what
you believe before seeing data; the LIKELIHOOD scores the data
under each hypothesis; the POSTERIOR is the renormalized
product -- belief after the evidence. On a grid of hypotheses
this is three arrays and two lines: `post = pr * lik;
post / reduce_add(post)`. Fully-Bayesian estimation keeps the
whole posterior instead of collapsing it to one number ([[MLE
vs MAP]]).

## Probability

A non-negative scalar that, with siblings, sums to 1.
Auto-tagged `Probability`; produced by `softmax`
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
today  but `quantize` does not.

## Quantization

Storing weights in low-precision integer formats (int8, int4)
for memory and speed at modest accuracy cost. Often combined
with LoRA in [[QLoRA]] workflows. MLPL stores everything in f64;
quantization is not in MLPL.

## Query (Q)

One of the three projections in attention. Each token emits
a query asking "what am I looking for?"; the dot product `Q
@ K^T` measures how strongly each query matches each key. In
MLPL: `Q = matmul(X, Wq)` where `Wq` is `[d_model, d_model]`
for single-head or `[d_model, d_k]` per head. Paired with
[[Key (K)]] and [[Value (V)]].

## RAG (Retrieval-Augmented Generation)

Fetch relevant documents from a corpus at query time, prepend
them to the prompt, and let an LLM answer over the retrieved
context. Reduces hallucination and lets you cite. MLPL ships
`pairwise_sqdist` / `knn` for similarity search; full RAG
pipelines are not in MLPL.

## randn / random (builtins)

`randn(seed, [shape...])` returns a standard-normal
sample (mean 0, variance 1) with the given shape. `random(
seed, [shape...])` returns uniform `[0, 1)` samples. Both
deterministic given the same seed.

## Rank (of a tensor)

The number of dimensions of a tensor. A scalar has rank 0; a
vector rank 1; a matrix rank 2; a `[batch, time, dim]`
tensor rank 3. MLPL: `rank(x)` returns the count;
`shape(x)` returns the dim sizes. Matches NumPy `.ndim` /
PyTorch `.dim()`. Distinct from "rank" in
linear algebra (the dimension of a matrix's column span),
from "low-rank" in LoRA (the small inner dimension of
the adapter matrices A and B), and from `depth`, which
counts levels of nesting rather than axes.

## Engram

A conditional-memory architecture (DeepSeek lineage): a trainable
n-gram hash table bolted into selected Transformer residual
blocks, so frequent local patterns are RETRIEVED from a table
instead of recomputed by attention/MLP capacity. The pipeline is
deterministic addressing -- normalize token ids, hash rolling
bigrams/trigrams per independent head (`ngram_hash`), gather
low-dimensional rows from one flattened table (`gather_rows`) --
followed by learned use: project the retrieved vectors, gate them
against the current hidden state, and add the result into the
residual stream. sw-MLPL builds it as composable language
primitives first (this glossary's `ngram_hash` / `gather_rows`),
then as the `engram(...)` model layer applied with
[[apply_engram]] -- differentiable end to end, so `train`/`adam`
scatter-ADD gradients into exactly the addressed table rows.

## engram (builtin)

`engram(hidden, ngrams, heads, slots, head_dim, seed)` builds a
trainable conditional-memory layer: one flattened `[rows,
head_dim]` table plus a value projection and a concat gate,
initialized NEAR-IDENTITY (zero table, zero value bias, gate bias
-2), so an untrained engram is an exact no-op on the residual
stream. Its five parameters are registered like any model's, so
`adam(loss, e, ...)` trains them. See [[Engram]] for the
architecture and [[apply_engram]] for the forward pass.

## apply_engram (builtin)

`apply_engram(e, h, ids)` runs the [[Engram]] forward pass: hash
`ids` with the layer's frozen spec, gather the addressed memory
rows, project them to a value vector `v`, then gate --
`out = h + sigmoid([h|v] @ Wg + bg) * v`. Differentiable: inside
`grad`/`train` the row gather lowers to a one-hot selection
matmul whose backward pass is a scatter-ADD, so only addressed
rows receive gradient and duplicate addresses accumulate. Unseen
token streams therefore keep their table rows at exactly zero.

## ngram_hash (builtin)

`ngram_hash(ids, orders, heads, slots, seed)` -- rolling n-gram
hash indices for [[Engram]]-style memory lookup: for every token
position it hashes the current token together with its n-1
predecessors (missing history pads with id 0) once per n-gram
order and per independent head, yielding a rank-3 `[T, order,
head]` array of table slot indices, each `< slots`. The
arithmetic is a FROZEN cross-backend contract (multiply mod a
fixed prime, sum, mod slots -- every intermediate exact in f64),
so CPU, MLX, and CUDA implementations must produce bit-identical
indices; token ids are capped at 2^21 - 1 to guarantee it. See
also: `gather_rows`.

## gather_rows (builtin)

`gather_rows(table, indices)` -- select whole rows of a rank-2
table by index: output shape is the indices' shape with the row
width appended, so `gather_rows(T, [[3, 0], [1, 1]])` on a
`[rows, d]` table yields `[2, 2, d]`. Out-of-range indices are a
loud error. The lookup half of [[Engram]] memory (hash with
`ngram_hash`, then gather the addressed rows from one flattened
table), and generally useful wherever embedding-style row
addressing is needed.

## :reset (REPL command)

Cancels ALL in-flight work on the connected backend (the
recovery move for a hung or slow run), after a y/N
confirmation. No-op in local browser mode.

## rotate (builtin)

`rotate(x, k, axis)` -- cyclic shift along an axis, APL's rotate. Positive k brings element k to the front (a left/up shift); negative k -- spelled `0 - k`, MLPL has no unary minus -- rotates the other way; any magnitude wraps. A pure permutation, so it is tape-differentiable (the gradient is the inverse rotation) and shape- and label-preserving. The workhorse of stencil-style neighborhoods: all 8 [[Game of Life]] neighbor shifts are rotate calls, and a permutation MATRIX is just `rotate(one_hot(iota(n), n), k, 0)` -- a rotated identity.

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
record-literal syntax . Field access is
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
Vision [[Transformer]] track wants
`load_preloaded("pets_tiny")` to return
`{X: [200, 3, 64, 64], Y: [200], names: [str]}` -- one builtin,
three logical outputs, no positional-tuple awkwardness.

Out of scope for the initial step: record destructuring in
let-bindings (`let {X, Y} = r`), record-update / spread syntax
(`{..r, X: new_x}`), pattern matching on records. Each is a
separate follow-up if a use case appears.

## Recursion

A function calling itself. MLPL user-defined functions
support recursion: `def u:fib(n) { if gt(n, 1) {
u:fib(n - 1) + u:fib(n - 2) } else { n } }`. The function
name is bound before the body executes, so self-reference
works. No tail-call optimization -- deep recursion will
hit a stack limit.

## Reconstruction Error

The difference between an [[Autoencoder]]'s input and its
output. Typically measured as mean squared error (MSE):
`reduce_add((output - input) * (output - input), 0) / N`.
Lower is better -- zero means perfect reconstruction.
Training minimizes this loss, forcing the [[Bottleneck
(autoencoder)]] to retain essential structure.

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
are not in MLPL.

## Representation Learning

The umbrella term for learning useful internal features
without explicit feature engineering. Self-supervised
pretraining, autoencoders, contrastive learning all fall
under it. embedding-visualization tools poke at
representations; full self-supervised pretraining is
not in MLPL.

## repeat block (language keyword)

`repeat N { body }` runs the body `N` times with no per-
iteration index binding. Ancestor of `train { ... }` (which
DOES bind `step` and capture loss). Use `repeat` for
iterative algorithms (k-means, [[power iteration]], MLP forward
+ backward demo) where you want a counted loop without the
training-specific bookkeeping.

## reshape (builtin)

`reshape(x, [d0, d1, ...])` returns a view of `x` with the
given dim sizes. Total element count must match; otherwise
`ShapeMismatch`. Clears axis labels (semantic identity is
lost on shape reflow); `reshape_labeled(x, dims, labels)`
preserves them by re-stating labels explicitly. Note: also
clears ValueTags, since the result no longer
represents the same domain.

## Residual

`y = x + f(x)`. A skip connection that lets gradients flow
through deep stacks. MLPL: `residual(inner_model)`.

## Result type

A `Value::Result { ok: bool, payload: Box<Value> }` wrapper
for ops that can fail without crashing the REPL. `ok(v)`
constructs `Ok(v)`; `err(v)` constructs
`Err(v)` -- typically `err("message")` but any Value
variant is allowed as the payload. The discriminator is a
bool, not a tag string, so `is_ok`/`is_err` are O(1) reads.

Accessors: `is_ok(r)` returns `1.0` / `0.0`, `is_err(r)` is
the inverse, `unwrap(r)` returns the payload if Ok else
raises `EvalError::UnwrapOnErr { message }` carrying the
payload's display form, `err_message(r)` returns the
payload if Err (else raises `Unsupported` -- Ok carries no
message), and `unwrap_or(r, default)` returns the payload
if Ok else evaluates and returns `default`. All accessors
raise `EvalError::NotAResult { receiver_kind, accessor }`
on a non-Result first argument.

Motivating use: the upcoming `:upload x` REPL command
 binds `x = Ok(image)` on a successful
upload or `x = Err("cancelled")` when the user dismisses
the file picker, so the program can branch on
`is_ok(x)` rather than getting tripped by an undefined
variable. The same shape generalizes to any future
fallible builtin: parse, file open, fetch, etc.

## Reward Hacking

When an RL agent finds a strategy that maximizes the reward
signal without solving the intended task -- exploiting bugs
in the reward function or environment. [[Goodhart's Law]] applied
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

## Hidden State

The fixed-size vector a recurrent network carries from one
time step to the next. It is the network's "memory" of
everything it has seen so far. At each step the cell reads
one new input and produces an updated hidden state. In MLPL,
`rnn_cell` and `lstm_cell` both accept and return hidden
state vectors. See also: RNN, LSTM, Vanishing Gradient.

## RNN (Recurrent Neural Network)

A network that processes sequences one element at a time,
updating a hidden state at each step:
`h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + bias)`. The
same weights are reused at every step (weight sharing).
MLPL: `rnn_cell(input, hidden, W_ih, W_hh, bias)` computes
one step; unroll in user code for a full sequence.

## LSTM (Long Short-Term Memory)

An RNN variant with gated memory that solves the vanishing
gradient problem. Adds a cell state alongside the hidden
state, controlled by three gates: forget (what to discard),
input (what to store), and output (what to expose). MLPL:
`lstm_cell(input, hidden, cell, W, bias)` returns a
concatenated `[hidden; cell]` vector. Split with
`reshape` + `take` to extract each.

## Vanishing Gradient

The problem that causes vanilla RNNs to forget early inputs
in long sequences. Because the hidden state passes through
`tanh` at every step, gradients shrink exponentially during
backpropagation through time. After ~10-20 steps the
gradient signal from early inputs is effectively zero, so
the network cannot learn long-range dependencies. LSTM and
GRU solve this with gated shortcuts that let gradients
flow unchanged across many steps.

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
positional encoding; RoPE is not implemented.

## Sampling

Drawing a random outcome from a probability distribution.
Multinomial sampling from logits: `sample(logits,
temperature, seed)`.

## Scaled dot-product attention

The canonical attention formula: `softmax(Q @ K^T / sqrt(d_k),
1) @ V`. The `sqrt(d_k)` divisor keeps the score variance
bounded as the key dimension grows so softmax doesn't
saturate into one-hot. Demos: "[[Attention]] Pattern" (heatmap
of weights only) and "Self-Attention from Scratch" (full
pipeline including `weights @ V`). [[Multi-head attention]] runs
this formula in parallel on `d_k = d_model / heads`-wide
slabs, then concatenates the per-head outputs.

## Scope (variable)

The region of code where a variable name is visible. MLPL
uses lexical scoping: a function body can read variables
from the enclosing scope, but function parameters shadow
outer variables only during the call and restore after.
Builtin names are global and cannot be shadowed by `def`.

## Scaling Laws

Empirical regularities relating model performance to model
parameters, dataset tokens, and compute budget -- typically
power laws with predictable exponents. Drove the "just make
it bigger" era of foundation models. The Compute-Optimality
Hypothesis (Chinchilla scaling) is a refinement: at a fixed
compute budget, smaller models trained on more data beat
bigger models trained on less.

## Segmentation

Classifying every pixel instead of the whole image: the output
is a mask the same shape as the input. [[U-Net]] is the
canonical architecture (contract to see context, expand to
localize, skip-connect to keep detail). Per-pixel
cross-entropy or Dice loss scores the overlap. At whiteboard
scale a segmentation task is just a classifier applied at every
grid cell.

## Self-attention

[[Attention]] where the queries, keys, and values all come from
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

## scan (higher-order)

`scan(:op, x)` -- the cumulative version of `reduce`. Returns
a same-shape array where each entry is the reduction over the
prefix up to that point. Standard for cumulative sum / running
max / prefix product. Not an MLPL builtin; `running_product`
covers the product case.

## scatter (builtin)

`scatter(buf, idx, value)` returns a copy of a rank-1 buffer
with the entry at `idx` replaced by `value`. neural-
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
render as `[batch=4, vocab=8]` instead of `[4, 8]`. See also
`size` / `tally` (element and major-cell counts), `depth`,
and `disp`.

## size / tally (builtins)

`size(x)` returns the total element count (numel) as a
scalar -- the product of the shape. `tally(x)` returns the
length of the leading axis (the number of major cells,
APL's monadic tally, J's `#`): a scalar tallies to `1`, and a
rank >= 1 array tallies to `shape[0]`. For a `[2, 3]` matrix
`size` is `6` while `tally` is `2`. Both round out the
structural-introspection set with `shape`, `rank`, `depth`,
and `disp`. Tensor mapping: `size` is NumPy's `.size` /
PyTorch's `.numel()` (element count) -- but beware, PyTorch's
`.size()` and MATLAB's `size()` return the SHAPE instead;
`tally` is `len(x)` / `x.shape[0]`, usually the batch size or
row count.

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
`sinkhorn_normalize`; compose it from `exp`, reductions, and
elementwise division.

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
temperature, axis)`. [[Temperature]] spreads probability mass
across more classes, so the targets carry richer
information than a one-hot label about which alternatives
the teacher considered. Training a student against these
softer targets via [[KL divergence]] is the core of knowledge
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
are not in MLPL.

## sqrt (builtin)

Elementwise square root: `sqrt(x)`. Used in attention
scoring (`/ sqrt(d_k)`), RMS norm, and PCA's Gram-Schmidt
normalization. Negative inputs produce NaN; the runtime does
not raise.

## Stack (tape op)

N-way concatenation of identically-shaped tensors along an
existing axis, lowered as a single tape node with N parents.
Used internally by the [[Multi-head attention]] tape lowering
(per-head outputs stacked along the column axis) and the
rank-3 batched attention path (per-batch outputs stacked
along the batch axis). Replaces the prior chained binary
[[concat (builtin)]], which had O(N^2) cost in both forward
and backward; stack is O(N).

Backward splits the upstream gradient into N equal-size
slabs and routes each to its parent. Same value-equivalence
as a left-associated `concat` chain, so analytic gradients
match the chain's exactly.

## Step

One optimizer update. The training loop runs `train N { body
}` for `N` steps; each step computes a forward pass, a
backward pass, and one parameter update. Distinct from
"epoch" (a full dataset pass).

## String list

A `Value::StrList { items: Vec<String> }` value built from a
`["a", "b", "c"]` literal whose every element evaluates to a
string . The same `[...]` surface syntax
dispatches on element kind: all strings -> `StrList`; all
numbers -> the existing `DenseArray` numeric path; mixed
kinds -> `EvalError::MixedArrayLitElements { kinds }` so the
user sees which positions disagreed. Empty `[]` continues to
produce an empty `DenseArray` for back-compat.

Use case: the Vision [[Transformer]] track wants
`load_preloaded("pets_tiny")` to return
`{X: [200, 3, 64, 64], Y: [200], names: ["Abyssinian_1.jpg",
"beagle_3.jpg", ...]}` -- one record value with three logical
outputs, including a per-image basename list, without
shoehorning the names into an index-keyed sub-record.

Today's accessors are minimal: `list_len(xs)` returns the
length as a scalar `DenseArray`. Indexing (`names[i]` or
`index(names, i)`) and iteration are not supported.

## Stop gradient / detach

Severing the autograd tape so gradients do not flow through
a value. Useful for target networks, EMA teachers, and
"freeze the encoder" patterns. MLPL does not ship
`stop_gradient` / `detach`;
`freeze(model)` covers the optimizer-side equivalent for
parameters.

## Superposition

A mechanistic-interpretability finding: neural networks pack
many more "concept directions" into their hidden activations
than there are neurons, by representing them as overlapping
directions in feature space. Explains why a single neuron
rarely encodes one clean concept and motivates dictionary-
learning-style decomposition.

## :status (REPL command)

Reports the connected backend(s): devices, GPUs, and live CPU /
RAM / GPU / VRAM readings. `:status watch` keeps the report
updating. In local browser mode it reports the WASM device and
how to connect a server.

## :tags / :untag (REPL commands)

Typed-value introspection. `:tags` lists every
binding with an attached `ValueTag` sorted alphabetically,
showing the tag's display form (e.g. `Probability`,
`Loss(CrossEntropy)`, `Weight(layer=linear_0, name=W)`).
`:untag <name>` clears the auto-tag from a binding when the
auto-tagger guessed wrong.

## :trace (REPL command)

`:trace on` / `:trace off` toggle execution tracing; a bare
`:trace` prints a summary of the last trace; `:trace json
[file]` prints the last trace as JSON or writes it to a file.

## tanh_layer / relu_layer / softmax_layer (builtins)

Parameter-free activation layers wrappable in a `chain(...)`.
`tanh_layer()` / `relu_layer()` / `softmax_layer()` apply
their respective elementwise functions. Distinct from the
math primitives (`tanh_fn`, `sigmoid`) in that layers can
participate in `apply(model, X)` and structural-
tail tagging.

## Supervised learning

The classical paradigm: training on (input, label) pairs
where a human (or curation pipeline) supplied the labels.
The model minimizes a per-example loss against the label
-- cross_entropy for classification, MSE for regression.
MLPL: every classifier demo (Logistic Regression, Tiny
MLP, Moons MLP, [[Softmax]] [[Classifier]]) is supervised. Cheap
and well-understood, but bounded by the supply of
labeled data.

## SVD (Singular Value Decomposition)

Any matrix factored as rotate-scale-rotate: `A = U S Vt` with
orthogonal U, V and non-negative singular values on S's
diagonal. Truncating to the top-k singular values gives the
best rank-k approximation -- the theory under [[PCA (Principal
Component Analysis)]], low-rank compression, and [[LoRA
(Low-Rank Adaptation)]]'s premise that weight UPDATES are
nearly low-rank. Not an MLPL builtin; PCA covers the common
use.

## SVM (Support Vector Machine)

A binary classifier that finds the maximum-margin
hyperplane separating two classes. The kernel trick lets
it implicitly use higher-dimensional spaces (RBF,
polynomial) so non-linearly-separable data becomes
separable in a transformed space. The dominant pre-deep-
learning classifier on small / tabular tasks; mostly
historical now. **Deferred** in MLPL: needs a quadratic-
program solver (or SMO algorithm).

## Swiss roll

A synthetic 2-D manifold (a rectangular sheet of paper) rolled
up like a Swiss-roll cake and embedded in 3-D space, then
projected back to 2-D as a dimensionality-reduction test. The
canonical "PCA fails, manifold methods win" benchmark: PCA's
linear axes slice through the roll, producing a smeared 2-D
shadow that crosses itself; [[t-SNE]] and [[UMAP]] (and Isomap)
recover the original 2-D rectangle by following the rolled
surface via local k-NN neighborhoods. MLPL's `umap_vs_pca`
demo uses two-moons embedded in higher D instead of a true
Swiss roll because the language does not yet have `sin` / `cos`
builtins (Swiss-roll construction needs them); when those land
the demo can swap in the textbook fixture.

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

## Tab completion (REPL)

See [[Completion popup (REPL)]]. The original step-043 MVP
used `Tab` as the trigger; step 046 settled on `Ctrl+Space`
(via a brief `Shift+Space` detour in step 045) because
`Tab` is reserved for browser element navigation.

## Tanh

`(exp(x) - exp(-x)) / (exp(x) + exp(-x))`. Squashes any real
number to `[-1, 1]`. Older alternative to [[ReLU]]; still
useful in small MLPs. MLPL: `tanh_layer()` or `tanh_fn`.

## to_number / to_int (builtins)

`to_number(s)` parses the string `s` as an `f64`;
`to_int(s)` parses it as an `i64` and rejects non-integer
numeric strings. Both return a [[Result type]] so the
caller branches explicitly on failure:

- `to_number("42")` -> `Ok(42)`
- `to_number("3.14")` -> `Ok(3.14)`
- `to_number("abc")` -> `Err("to_number: cannot parse \"abc\" as a number")`
- `to_int("42")` -> `Ok(42)`
- `to_int("3.5")` -> `Err("to_int: \"3.5\" is not an integer")`
- `to_int("xyz")` -> `Err("to_int: cannot parse \"xyz\" as an integer")`

Leading and trailing whitespace are trimmed before
parsing. Pair with [[args (builtin)]] to convert CLI
string arguments into numeric form, e.g.
`epochs = unwrap(to_int(arg(0)))`.

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
calling protocols are not in MLPL.

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

`take(x, axis, idx)` drops one axis at a single integer
index. For a rank-`r` input, the result has rank
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
exploration" lesson. Compare with [[UMAP]], which preserves
global inter-cluster distance in addition to local structure
and is the recommended modern default.

## UMAP

Uniform Manifold Approximation and Projection (McInnes and
Healy, 2018). A non-linear dimensionality reduction method that
projects high-D points down to 2-D (or 3-D) while preserving
*both* local neighborhood structure (like [[t-SNE]]) AND global
inter-cluster distances (unlike [[t-SNE]], which tends to
inflate well-separated clusters). The intuition is
Riemannian-geometric: assume the data lies on a smooth manifold
of unknown curvature, locally approximate that manifold by
fuzzy simplicial sets (per-point neighborhood graphs whose edge
weights are fuzzy-set memberships), then find a low-D
embedding whose fuzzy graph is as close as possible to the
high-D one in cross-entropy. The optimization is stochastic
gradient descent with negative sampling -- repulsive force is
estimated per step from a random subset of non-neighbor pairs
rather than the O(N^2) all-pairs sum t-SNE pays. MLPL:
`umap(X, n_neighbors, min_dist, iters, seed)` returns `[N, 2]`.
`n_neighbors` controls how much local vs global structure is
weighted (typical: 10-30); `min_dist` is a soft floor on
attractive distances (typical: 0.1-0.5). UMAP is the
recommended default for visualizing learned embeddings; it
underpins the comparison demos in the dimensionality-reduction
milestone (see `demos/umap_vs_pca.mlpl` and
`demos/umap_vs_tsne.mlpl`).

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
is a tiny taste; full uncertainty tooling is not in MLPL.

## unfreeze (builtin)

`unfreeze(m)` is the inverse of `freeze(m)` -- removes every
param of `m` from the env's frozen set so subsequent
`adam` / `momentum_sgd` updates can move them again. Used
together as the LoRA freeze / unfreeze pair.

## Universal Approximation

Theorem: a feed-forward network with a single hidden layer
can approximate any continuous function on a compact domain
to arbitrary precision, given enough hidden units. Existence
result, not a learnability claim. Justifies why neural nets
are not architecturally limited; says nothing about whether
gradient descent will find the right weights.

## Unsupervised learning

Training without labels: the model discovers structure in
the data itself. [[Clustering]], dimensionality reduction,
density estimation. MLPL: the [[K-Means]] demo, the PCA demo,
and [[t-SNE]] (`tsne(X, perp, iters, seed)`) are unsupervised
-- they group / project points using only geometry.
[[Self-supervised learning]] is a closely-related modern
variant where labels are *derived* from the input rather
than absent.

## :upload (REPL command)

`:upload <name>` (web REPL only) opens the browser's file
picker. Pick any image the browser can decode (PNG, JPEG,
WebP, etc.) and the Canvas API resizes it to 64x64,
normalizes the RGB pixels into the range `[-1, 1]`, and
binds the result under your chosen name as a `Result`
value. On success you get `<name> = Ok({pixels: [1, 3, 64,
64], h: 64, w: 64})`, ready to feed straight into a
trained ViT classifier. On dismiss you get `<name> =
Err("cancelled")`. Bad-format files (a binary renamed
`.jpg`) bind `Err("decode failed: not a valid image")`;
unreadable files bind `Err("read failed")`.

Inspect with `is_ok(<name>)`, `is_err(<name>)`, or read
the specific message with `err_message(<name>)`. Get the
tensor out with `unwrap(<name>).pixels`. View it with
`svg(unwrap(<name>).pixels, "gallery")`. The legacy
"Upload Image" button uses the same pipeline but writes
to a hardcoded variable named `uploaded`.

## User-Defined Function (UDF)

A function written in MLPL with `def ns:name(params) { body }`.
Names require a colon namespace prefix: `u:` for end users,
`vendor:` for packages. Builtin names (no colon) cannot be
shadowed. The last expression in the body is the return
value; use `return expr` for early exit. Functions support
recursion and read (but do not mutate) outer variables.
MLPL: `def u:double(x) { x * 2 }` then `u:double(21)`.

## VAE (Variational Autoencoder)

An autoencoder where the latent space is regularized to
match a prior distribution (typically Gaussian). The encoder
outputs `(mean, std)`; the decoder samples from
`Gaussian(mean, std)`. Trained with reconstruction + KL-
divergence terms. plans first-class
`Distribution` support; VAE demos follow once
distributions ship.

## :vars (REPL command)

`:vars` lists every bound array variable with its shape
(labeled if any axes are named) and ValueTag if any.
Trainable params are flagged `[param]`; frozen params show
in `:wsid`'s frozen-count.

## :version (REPL command)

Prints the sw-MLPL version and the target architecture the REPL
was built for.

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
[[Query (Q)]] and [[Key (K)]].

## ViT (Vision Transformer)

Apply a transformer directly to images by splitting the
image into fixed-size patches (e.g. 16x16 pixels),
flattening each patch into a vector, and treating the
sequence of patch vectors as tokens (with positional
embeddings). Showed that the inductive biases of CNNs are
not strictly required at scale -- pure [[Attention]] works
on images too. Foundation for modern vision-language
models.

Shipped in MLPL via the [[patchify (builtin)]] builtin
(image -> patch tokens), [[concat (builtin)]] (prepend
CLS), and [[Multi-head attention]] on the autograd tape
with the [[Stack (tape op)]] primitive to keep batched
and multi-head paths O(N) rather than O(N^2). The web
playground's "Pets: cat vs dog (quick)" and "Pets: multi-
head ViT (quick + viz)" demos train a tiny ViT on the
[[Oxford-IIIT Pet dataset]]; the [[:upload (REPL command)]]
command lets you classify your own photo against the
trained model.

## VLM (Vision-Language Models)

Models that take both image and text inputs (e.g. CLIP,
LLaVA, GPT-4V). Out of MLPL's current scope -- the language
core is text + tabular today; vision pipelines are a
follow-up step.

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
in `Attention`). Auto-tagged `Weight(layer, name)`.

## :wsid (REPL command)

`:wsid` (workspace ID) prints summary counts: variables,
trainable parameters, frozen parameters, models, tokenizers,
optimizer slots, experiment records. Inspired by APL's
`)WSID`. The first command to run when you reopen a
session and want a quick sense of state.

## Weight Decay

A regularization technique that shrinks weights toward zero
on each update step (L2 penalty). Often baked into the
optimizer (`AdamW`). MLPL expresses it as one extra loss term
-- `loss + lam * sum(W * W)` -- as the "Taming Overfitting:
Weight Decay" demo shows; optimizer-side decoupled decay
(AdamW-style) is not an MLPL builtin. The L1 variant
(`lam * sum(abs(W))`) prefers exact zeros -- sparsity -- where
L2 prefers many small weights.

## Weight Initialization

The scheme that sets parameter values before training: zeros
(biases), small Gaussian (general weights), Xavier / Glorot
(scaled by fan-in for tanh / sigmoid), He / Kaiming (scaled
for [[ReLU]]). MLPL initializes weights via `randn(seed,
shape)` scaled by 0.5 inside `linear`; explicit
Xavier / He variants are not in MLPL.

## Workspace

The full set of bindings and metadata in the current REPL
session: vars, params, models, tokenizers, optimizer state,
experiment log, frozen set, tag side-table, peer
dispatcher. `:wsid` shows the counts; `:vars`, `:models`,
`:tags`, `:experiments` list contents.

## World Model

A learned model of an ENVIRONMENT's dynamics -- given state
and action, predict the next state (often in latent space, see
[[JEPA (Joint-Embedding Predictive Architecture)]]) -- so an
agent can plan by imagination instead of trial. Not in MLPL.


## XOR (not linearly separable)

XOR (exclusive or): the four 2-input boolean cases
`[0,0] -> 0`, `[0,1] -> 1`, `[1,0] -> 1`, `[1,1] -> 0`.
Positives sit on one diagonal, negatives on the other. No
straight line in input space can separate them, so a linear
classifier (weight vector + [[Sigmoid]]) cannot learn XOR.
Adding a single hidden layer with a nonlinear activation
([[Tanh]] or [[ReLU]]) makes XOR learnable -- the hidden layer creates intermediate
features along each diagonal that the output layer can
linearly combine.

The web playground walks both cases as paired demos. The
linear failure case is the fifth part of "Decision
Boundary: logical gates" (AND, OR, NAND, NOR all succeed;
XOR's surface stalls near the uniform 0.5 prior). The
working version is "Decision Boundary: XOR (with MLP)" --
load it from the demo dropdown after the gates demo to see
the curved boundary a hidden layer produces.

Historically this is the result Minsky and Papert called
out in 1969 to argue against shallow perceptrons; the
multi-layer answer didn't take off until backpropagation
made it tractable in the mid-1980s.

## FP32

32-bit floating point (4 bytes/number): 1 sign bit, 8
exponent bits, 23 mantissa bits. Full precision -- the
default for training and the baseline that [[Quantization]]
is measured against. See [[Exponent and mantissa]].

## FP16

16-bit "half" floating point (2 bytes): 1 sign, 5 exponent,
10 mantissa. Half the size of [[FP32]] with good precision,
but the small 5-bit exponent gives a narrow range (~6e-5 to
65504) so large activations can overflow to infinity. Often
written `F16` in GGUF files. Contrast [[BF16]].

## BF16

bfloat16 (2 bytes): 1 sign, 8 exponent, 7 mantissa. Same
size as [[FP16]] but spends bits differently -- the full
8-bit exponent gives [[FP32]]'s range with coarser
precision. The modern training default, because training
tolerates low precision better than overflow. See
[[Exponent and mantissa]].

## FP8

8-bit floating point (1 byte), in two flavors: `E4M3` (4
exponent, 3 mantissa -- more precision) and `E5M2` (more
range). Emerging for inference and training on the newest
hardware. Smaller than INT8 quant in bytes but a true float
format, not a scaled integer. See [[Scale (quantization)]].

## Exponent and mantissa

The two number-carrying fields of a floating-point value
(besides the sign bit). The **exponent** sets the range (how
big or small); the **mantissa** sets the precision (how many
significant digits). [[FP16]] and [[BF16]] are the same size
but split these differently -- FP16 favors mantissa
(precision), BF16 favors exponent (range).

## Quantization error

The gap between an original weight and its
[[Quantization]]-then-dequantized value. For
round-to-nearest it is bounded by half a scale step. Error
grows gently from 8-bit to 4-bit, then climbs steeply below
4-bit -- the "cliff" that makes sub-4-bit quantization hard.
See [[Scale (quantization)]].

## Bits per weight

The average number of bits used to store one weight (bpw),
including the overhead of block scales. It sets a quantized
model's file size: `size ~= params * bpw / 8`. Examples:
`Q8_0` ~8.5, `Q6_K` ~6.6, [[Q4_K_M]] ~4.85, `IQ2_M` ~2.7,
ternary ~1.58. See [[BitNet b1.58 (1.58-bit)]].

## Scale (quantization)

The shared floating-point multiplier that maps stored
integers back to real values: `value ~= integer * scale`.
For symmetric per-tensor quant, `scale = max(abs(W)) / qmax`
(e.g. `qmax = 127` for signed 8-bit). The single number that
makes integer [[Quantization]] reversible. See [[Zero-point]].

## Zero-point

An integer offset added to scaled values so the representable
range can sit off-center: `value ~= (integer - zero_point) *
scale`. Used by **asymmetric** quantization for lopsided data
(e.g. all-positive post-ReLU activations). See [[Symmetric
and asymmetric quantization]] and [[Scale (quantization)]].

## Symmetric and asymmetric quantization

Two ways to place the integer grid. **Symmetric** centers it
on zero and uses a scale only (GGUF `_0` types).
**Asymmetric** adds a [[Zero-point]] so the grid can shift to
fit lopsided data (GGUF `_1` types). Asymmetric fits the data
better at the cost of storing one extra number per block.

## Per-tensor, per-channel, and per-block quantization

The granularity of the [[Scale (quantization)]].
**Per-tensor** uses one scale for a whole weight matrix --
cheapest, but one outlier inflates the scale and wastes range
on everyone else. **Per-channel** uses one scale per row;
**per-block** uses one per group of 32-256 weights. Finer
granularity spends a few extra bits on scales to cut
[[Quantization error]] -- the idea [[K-quant]] is built on.

## Activation outlier

A few channels of a transformer's activations whose
magnitudes are far larger than the rest. They break naive
INT8 quantization (they dominate the scale); [[LLM.int8()]]
handles them by keeping those channels in [[FP16]]. The
reason weight quantization is easier than activation
quantization.

## Ternary weights

Weights restricted to three values: -1, 0, +1. Three states
carry `log2(3) ~= 1.58` bits, so ternary is "1.58-bit". The 0
lets a weight switch off (sparsity), which is why ternary
beats pure binary. See [[BitNet b1.58 (1.58-bit)]].

## BitNet b1.58 (1.58-bit)

A model trained from scratch with [[Ternary weights]] (-1/0/
+1), so it learns to work within the constraint rather than
being quantized after the fact. At scale it can match an
[[FP16]] baseline at ~1.58 [[Bits per weight]] -- the extreme
end of [[Quantization-aware training (QAT)]] (Ma et al.,
2024).

## GGUF

The file format used by llama.cpp to ship quantized models.
A GGUF file's name encodes its quantization type -- e.g.
[[Q4_K_M]], `IQ2_XXS`, `Q8_0` -- which tells you the
bit-width, method, and size tier. See [[K-quant]], [[i-quant
(IQ)]], and [[Quant size modifiers (S, M, L)]].

## K-quant

The modern integer-quantization scheme in [[GGUF]] (the `_K`
types, `Q2_K` through `Q6_K`). A [[Super-block]] of 256
weights is split into sub-blocks, each with its own quantized
scale, plus a super-block scale on top -- so bits are spent
where the data needs them. Better quality per bit than the
legacy `Q4_0`-style types. The `_K` in a name means k-quant.
See [[Legacy quants (Q4_0, Q4_1)]].

## Super-block

A block of 256 weights (in [[K-quant]]) subdivided into
smaller sub-blocks. The super-block carries one
floating-point scale; each sub-block carries its own low-bit
quantized scale. This hierarchy is how k-quants get fine
[[Per-tensor, per-channel, and per-block quantization]] cheaply.

## i-quant (IQ)

The lowest-bit [[GGUF]] family (`IQ1_*` through `IQ4_*`).
i-quants use an [[Importance matrix (imatrix)]] to protect
the weights that matter most, plus codebook / non-linear
encoding, which is what makes 2-3 bit usable where plain
rounding fails. The `IQ` prefix means importance-quant.

## Importance matrix (imatrix)

A table of per-weight (or per-channel) importance scores
computed by running calibration data through the model.
Low-bit [[i-quant (IQ)]] methods use it to decide which
weights to encode carefully and which to round hard -- the
key to sub-3-bit quality. Calibrating on data unlike real use
weakens it.

## Quant size modifiers (S, M, L)

The suffix on a [[K-quant]] name that picks a size/quality
tier by mixing precisions across tensors. `_S` = Small, `_M`
= Medium, `_L` = Large; bigger letter = more quality + more
size. [[Q4_K_M]] keeps most tensors at 4-bit but bumps a few
sensitive ones to 6-bit; [[Q4_K_S]] is more uniformly 4-bit.
See also [[Extra-small quant tiers (XS, XXS)]] and [[XL quant
tier]].

## Extra-small quant tiers (XS, XXS)

Size suffixes below `_S`, used mostly by [[i-quant (IQ)]]
types: `_XS` (extra-small) and `_XXS` (extra-extra-small),
e.g. `IQ2_XXS`, [[IQ4_XS]]. The smallest, most aggressive
tiers -- maximum compression, lowest quality. See [[Quant
size modifiers (S, M, L)]].

## XL quant tier

A size suffix above `_L` (extra-large), seen on [[Unsloth
Dynamic (UD-)]] types like `UD-Q4_K_XL`. Spends extra bits on
the most sensitive tensors for quality closer to the next
bit-width up, at a larger file size. See [[Quant size
modifiers (S, M, L)]].

## Unsloth Dynamic (UD-)

A `UD-` prefix (Unsloth's "Dynamic" quants) marks a GGUF that
quantizes different layers to different bit-widths using a
calibration analysis -- keeping more bits where the model is
sensitive and fewer where it is not. Examples: `UD-Q4_K_XL`,
`UD-IQ2_M`, `UD-Q2_K_XL`. A per-layer refinement of
[[K-quant]] / [[i-quant (IQ)]]; read the rest of the name
normally.

## Legacy quants (Q4_0, Q4_1)

The original [[GGUF]] integer scheme, before [[K-quant]]. A
block of 32 weights shares one scale (`_0`, symmetric) or a
scale plus a minimum (`_1`, asymmetric). Mostly superseded by
K and [[i-quant (IQ)]] at the same size -- prefer those --
though [[Q8_0]] remains common. See [[Symmetric and
asymmetric quantization]].

## NL (non-linear quant)

The `NL` suffix (as in [[IQ4_NL]]) marks a non-linear
quantization grid: the representable levels are spaced to
match the bell-curve distribution of weights instead of
evenly, cutting [[Quantization error]] for the same bit
count. Related idea: [[NF4 (NormalFloat 4-bit)]].

## Post-training quantization (PTQ)

Quantizing a model's weights AFTER training in floating
point, with no retraining. Cheap and fast; the default for
8/6/4-bit and what nearly every [[GGUF]] download is. The
model never saw quantization, so quality holds at 4+ bits but
degrades below. Contrast [[Quantization-aware training
(QAT)]].

## Quantization-aware training (QAT)

Training with the quantization rounding simulated in the
forward pass (a "fake quant"), so the model adapts its
weights to the coarse grid; gradients still flow in full
precision via the [[Straight-through estimator]]. More
expensive than [[Post-training quantization (PTQ)]] but holds
quality at very low bit-widths. [[BitNet b1.58 (1.58-bit)]]
is the extreme (train at the target precision).

## Straight-through estimator

The trick that makes [[Quantization-aware training (QAT)]]
work: the forward pass rounds (a step function with zero
gradient almost everywhere), but the backward pass pretends
the rounding was the identity, so gradients pass straight
through to the underlying full-precision weights.

## GPTQ

A one-shot 4-bit [[Post-training quantization (PTQ)]] method
(2022) that uses second-order (Hessian) information to
compensate, column by column, for the error each rounding
introduces. Made 4-bit LLMs practical without retraining.

## AWQ

Activation-aware Weight Quantization (2023): identify the
~1% of weight channels that matter most (by [[Activation
outlier]] magnitude) and protect them, quantizing
the rest to 4-bit. Strong low-bit quality with no backprop.

## QLoRA

Fine-tuning with the base model kept 4-bit ([[NF4 (NormalFloat
4-bit)]]) and frozen, training only small floating-point LoRA
adapters (2023). Put fine-tuning of large models on a single
GPU. Because the base stays quantized during training, there
is no train/inference precision mismatch on the base.

## NF4 (NormalFloat 4-bit)

A 4-bit data type whose 16 levels are placed to match the
bell-curve (normal) distribution of neural-network weights,
so each level is "used" equally often. Introduced with
[[QLoRA]]; a non-linear grid like [[IQ4_NL]]. See [[NL
(non-linear quant)]].

## LLM.int8()

An 8-bit [[Post-training quantization (PTQ)]] method (2022)
that fixed naive INT8's failure on large transformers: it
keeps the few [[Activation outlier]] channels in
[[FP16]] and quantizes the rest to INT8. The first robust
8-bit for LLMs.

## Q8_0

8-bit legacy/block quant (~8.5 [[Bits per weight]]). Near-
lossless -- the safe choice when you just want to halve
[[FP16]] with no visible quality loss. Still common despite
newer methods. See [[Legacy quants (Q4_0, Q4_1)]].

## Q6_K

6-bit [[K-quant]] (~6.6 [[Bits per weight]]). Very close to
[[FP16]] quality; the high-quality inference setting when you
have the memory. The next step down from [[Q8_0]] for far
less size.

## Q5_K_S

5-bit [[K-quant]], Small tier. A bit smaller and slightly
lower quality than [[Q5_K_M]]. See [[Quant size modifiers (S,
M, L)]].

## Q5_K_M

5-bit [[K-quant]], Medium tier (~5.5 [[Bits per weight]]). A
safe "quality" choice above [[Q4_K_M]] when memory allows.

## Q4_K_M

4-bit [[K-quant]], Medium tier (~4.85 [[Bits per weight]]).
The community default: roughly half of [[FP16]] with quality
loss most people never notice. Bumps a few sensitive tensors
to 6-bit. Start here; go up ([[Q5_K_M]] / [[Q6_K]]) for
quality.

## Q4_K_S

4-bit [[K-quant]], Small tier. More uniformly 4-bit than
[[Q4_K_M]] -- a little smaller and slightly lower quality.

## Q3_K_M

3-bit [[K-quant]], Medium tier. Noticeably lower quality than
4-bit; usable on large models, risky on small ones. Prefer an
[[i-quant (IQ)]] at this size if available.

## Q3_K_S

3-bit [[K-quant]], Small tier. Aggressive; quality drops
fast, especially on small models. Avoid below ~13B unless you
must. See [[Quant size modifiers (S, M, L)]].

## Q2_K

2-bit [[K-quant]] (~2.6 [[Bits per weight]]). The smallest
k-quant; quality is shaky and craters on small models. At
this size an [[i-quant (IQ)]] (`IQ2_M`, `IQ2_XS`) is usually
better.

## IQ4_XS

4-bit [[i-quant (IQ)]], extra-small tier (~4.25 [[Bits per
weight]]). Uses an [[Importance matrix (imatrix)]] to beat
[[Q4_K_S]] at a slightly smaller size -- a good low-4-bit
pick. See [[Extra-small quant tiers (XS, XXS)]].

## IQ4_NL

4-bit [[i-quant (IQ)]] with a [[NL (non-linear quant)]] grid. Similar size to legacy `Q4_0` but better
quality thanks to the bell-curve-matched levels; a drop-in
upgrade where `Q4_0` was used.

## IQ3_XXS

3-bit [[i-quant (IQ)]], extra-extra-small tier (~3.0 [[Bits
per weight]]). Very aggressive; relies on the [[Importance
matrix (imatrix)]] to stay usable. For fitting a big model in
little memory.

## IQ2_M

2-bit [[i-quant (IQ)]], Medium tier (~2.7 [[Bits per
weight]]). About the lowest bit-width that stays usable, and
only on large models with a good [[Importance matrix
(imatrix)]]. Beats [[Q2_K]] at a similar size.

## INT8

8-bit integer quantization: each weight stored as an integer
in `[-127, 127]` with a shared [[Scale (quantization)]].
~255 levels -- near-lossless for most models and the safe way
to halve [[FP16]]. The 8-bit rung of the [[Quantization]]
ladder; the GGUF form is [[Q8_0]]. A scaled integer, unlike
the floating-point [[FP8]]. See [[LLM.int8()]] for the
outlier issue on large transformers.

## INT4

4-bit integer quantization: ~15 levels. The popular sweet
spot -- a big size win with small quality loss, but only with
good methods ([[K-quant]], [[GPTQ]], [[AWQ]], an [[Importance
matrix (imatrix)]]); naive per-tensor 4-bit is rougher. GGUF
forms include [[Q4_K_M]] and [[IQ4_XS]]. See [[Bits per
weight]].

## IQ2

The 2-bit [[i-quant (IQ)]] family (`IQ2_XXS`, `IQ2_XS`,
`IQ2_S`, [[IQ2_M]]). About the lowest bit-width that stays
usable, and only on large models with a good [[Importance
matrix (imatrix)]]. At this size i-quants beat the
[[K-quant]] [[Q2_K]]. See [[Extra-small quant tiers (XS,
XXS)]].

## IQ3

The 3-bit [[i-quant (IQ)]] family (`IQ3_XXS`, `IQ3_S`,
`IQ3_M`). Uses an [[Importance matrix (imatrix)]] to stay
usable below 4-bit; a better choice than [[Q3_K_S]] /
[[Q3_K_M]] at the same size for fitting a big model in little
memory.

## Linear Regression

Fit y = W*x + b by minimizing squared error -- the simplest
supervised model and the doorway to everything else. In MLPL
there is no closed-form normal-equation builtin; you solve it
the way you solve every model here, with [[Gradient descent]]:
the "How Gradient Descent Works" demo IS linear regression,
fitting y = w*x + b while walking the [[Loss]] surface downhill.
Add a [[Sigmoid]] on top and you have [[Logistic Regression]].

## Logistic Regression

A linear model pushed through a [[Sigmoid]] so the output reads
as a probability, trained on cross-entropy loss -- the smallest
useful classifier and the discriminative counterpart to
[[Naive Bayes]]. Hands-on: the "Logistic Regression" demo and
the "Machine Learning: Logistic Regression" lesson train one on
moons data and draw its decision boundary; the two "Decision
Boundary" demos show what a single linear unit can and cannot
separate (XOR needs an [[MLP (Multi-Layer Perceptron)]]).

## k-Nearest Neighbors

Classify a point by majority vote of the k closest training
points -- no training phase at all; the distance matrix IS the
model. Deeply array-spirited: squared distances are
|a|^2 + |b|^2 - 2ab (one matmul plus two reductions), the
neighbor pick is top_k, the vote is one_hot + reduce_add +
argmax. MLPL ships the [[knn (builtin)]] for neighbor indices
(used in embedding inspection). Contrast
[[K-Means]], which clusters unlabeled data instead of voting
with labels.

## Name Forms (name / :name / u:name)

sw-MLPL's three deliberate name roles: bare `name(...)` CALLS a builtin; `:name` QUOTES it into a first-class value (what `reduce(:add, x)` consumes -- typing `:disp` shows the reference, calling `disp(G)` renders); `u:name` is the mandatory namespace for YOUR functions, so user code can never collide with present or future builtins (the cure for APL's workspace name-clash history, priced at two characters). Introspect each space with `:builtins`, `:fns`, `:list u:name` (verbatim source, comments included), and `:describe`. See "The Three Kinds of Name" in the Language Reference tab.

## Naive Bayes

A generative classifier: apply Bayes' rule with the "naive"
assumption that features are independent given the class. The
Gaussian variant needs only per-class feature means and
variances -- masked reductions in array terms -- and predicts by
argmax over summed log-densities. Fast, surprisingly strong on
small data, and the classic generative-vs-discriminative foil
to [[Logistic Regression]].

## Gradient Boosting

An ensemble built sequentially: each new weak learner (almost
always a shallow [[Decision Tree]]) is fit to the residual
errors of the ensemble so far, and predictions are the sum of
all learners. Where [[Random Forest]] reduces variance by
averaging independent trees, boosting reduces bias by stacking
corrections -- see [[Ensembling]] for the umbrella idea.
**Deferred** in MLPL: greedy tree fitting is control-flow-heavy
and non-differentiable, so it stays a glossary concept.

## XGBoost

"Extreme Gradient Boosting" -- an engineered implementation of
[[Gradient Boosting]] with a regularized objective,
second-order (Newton) update steps, histogram-based split
search, and native sparsity handling. The de-facto strong
baseline for tabular data. **Deferred** in MLPL: library-scale
engineering, not an array-language teaching target; read the
[[Decision Tree]] and [[Gradient Boosting]] entries for the
ideas it builds on.
