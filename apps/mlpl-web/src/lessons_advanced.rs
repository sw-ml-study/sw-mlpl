//! Saga 15 / 16 / 20 tutorial lesson data, extracted out
//! of `lessons.rs` to keep that file under its
//! sw-checklist LOC budget. Each advanced lesson is a
//! named const; `lessons::LESSONS` references them in
//! order.

use crate::lessons::Lesson;

/// Saga 20 tutorial lesson.
pub const NEURAL_THICKETS: Lesson = Lesson {
    title: "Neural Thickets",
    intro: "Saga 20 ships four builtins that compose into the Neural Thickets (RandOpt-style) workflow: clone_model(m) deep-copies a model with fresh param names, perturb_params(m, family, sigma, seed) adds Gaussian noise to a named family of params (all_layers / attention_only / mlp_only / embed_and_head), argtop_k(values, k) returns indices of the K best entries, and scatter(buffer, index, value) writes one scalar into a rank-1 buffer. The pattern: train a base, clone N times, perturb each clone along some family, score each on held-out tokens, argtop_k the specialists, ensemble. This lesson runs a tiny 4x4 sweep (V=8, d=4) so the whole heatmap renders in the browser; demos/neural_thicket.mlpl has the full Shakespeare version and docs/using-perturbation.md has the retrospective.",
    examples: &[
        "base = chain(embed(8, 4, 0), residual(chain(rms_norm(4), causal_attention(4, 1, 1))), residual(chain(rms_norm(4), linear(4, 8, 2), relu_layer(), linear(8, 4, 3))), rms_norm(4), linear(4, 8, 4))",
        "val_X = [1, 3, 5, 7, 2, 4, 6, 0]",
        "val_Y = [3, 5, 7, 2, 4, 6, 0, 1]",
        "v = clone_model(base)",
        "perturb_params(v, \"attention_only\", 0.05, 42)",
        "cross_entropy(apply(v, val_X), val_Y)",
        "sigma = 0.05",
        "losses = zeros([16])",
        "for i in [0, 1, 2, 3] { v = clone_model(base); perturb_params(v, \"all_layers\", sigma, i + 100); losses = scatter(losses, i, cross_entropy(apply(v, val_X), val_Y)) }",
        "for i in [0, 1, 2, 3] { v = clone_model(base); perturb_params(v, \"attention_only\", sigma, i + 200); losses = scatter(losses, 4 + i, cross_entropy(apply(v, val_X), val_Y)) }",
        "for i in [0, 1, 2, 3] { v = clone_model(base); perturb_params(v, \"mlp_only\", sigma, i + 300); losses = scatter(losses, 8 + i, cross_entropy(apply(v, val_X), val_Y)) }",
        "for i in [0, 1, 2, 3] { v = clone_model(base); perturb_params(v, \"embed_and_head\", sigma, i + 400); losses = scatter(losses, 12 + i, cross_entropy(apply(v, val_X), val_Y)) }",
        "heat = reshape(losses, [4, 4])",
        "svg(heat, \"heatmap\")",
        "best_idx = argtop_k(-1.0 * losses, 4)",
    ],
    try_it: "Rerun the sweep at sigma = 0.2 and at sigma = 0.01. How does the heatmap change? At large sigma, every family's loss blows up; at small sigma the heatmap stays close to base. Then try swapping apply(v, val_X) for apply(base, val_X) in one family to see the base's loss show up as a row.",
};

/// Saga 15 tutorial lesson.
pub const LORA_FINE_TUNING: Lesson = Lesson {
    title: "LoRA Fine-Tuning",
    intro: "Saga 15 ships three builtins for parameter-efficient fine-tuning: freeze(m) marks every param of m frozen (adam / momentum_sgd skip frozen names), unfreeze(m) is the inverse, and lora(m, rank, alpha, seed) wraps every Linear in m with two low-rank adapter matrices A [in, rank] and B [rank, out] and auto-freezes every non-adapter param in the returned student. Forward is y = X @ W + (alpha / rank) * X @ A @ B + b. B zero-inits so apply(lora_m, X) matches the base exactly before any gradient step; A inits as scaled randn so learning has somewhere to go. Only the adapters train. This lesson runs a tiny interactive version (V=8, d=4, rank=2) so the forward and the learned adapter render quickly in the browser; demos/lora_finetune.mlpl has the full Shakespeare version and docs/using-lora.md has the retrospective.",
    examples: &[
        "base = chain(embed(8, 4, 0), residual(chain(rms_norm(4), causal_attention(4, 1, 1))), linear(4, 8, 1))",
        "student = lora(base, 2, 4.0, 7)",
        "X = [1, 3, 5, 7, 2, 4, 6, 0] ; Y = [3, 5, 7, 2, 4, 6, 0, 1]",
        "train 10 { adam(cross_entropy(apply(student, X), Y), student, 0.05, 0.9, 0.999, 0.00000001); loss_metric = cross_entropy(apply(student, X), Y) }",
        "loss_curve(last_losses)",
        "cross_entropy(apply(student, X), Y)",
        "cross_entropy(apply(base, X), Y)",
    ],
    try_it: "The two cross_entropy lines after training should report different numbers: student's loss went down during fine-tune, but base's loss is unchanged because lora() auto-froze the base -- adam only moved the adapters. Try unfreeze(student) before the train block and re-run: now both losses move.",
};

/// Saga 23 tutorial lesson: typed ML values (Tier A vocabulary).
pub const TYPED_ML_VALUES: Lesson = Lesson {
    title: "Typed ML Values",
    intro: "Saga 23 ships a curated Tier A typed-value vocabulary -- Logit, Probability, LogProbability, Loss, Gradient, Weight, Bias, Activation, LearningRate, Labels, AttentionMap -- attached to bindings via a side table on Environment. The runtime auto-tags producers (softmax -> Probability, cross_entropy -> Loss, grad -> Gradient, linear -> Weight + Bias, apply on a Linear-tailed model -> Logit). Predicate consumers reject mismatched tags with EvalError::TypeMismatch carrying a 3-5 line tutoring hint. Tags propagate through arithmetic / transpose / reshape / reductions: Logit + Logit stays Logit, Loss survives mean/reduce_add, reshape clears, and domain-mixing combos like Logit + Probability raise a tutoring TypeMismatch. New REPL commands :tags lists every tagged binding and :untag clears one. Untyped programs keep working unchanged (gradual-typing additivity). This lesson walks the canonical pipeline -- logits to probs to loss to gradient to weight update -- and demos the canonical double-softmax bug now caught at the call site.",
    examples: &[
        "L = randn(0, [2, 3])",
        ":describe L",
        "probs = softmax(L, 1)",
        ":describe probs",
        "T = [0.0, 1.0]",
        "loss = cross_entropy(L, T)",
        ":describe loss",
        ":tags",
        "loss = cross_entropy(probs, T)",
        "lr = cosine_schedule(0, 100, 0.001, 0.01)",
        ":describe lr",
        "W = param[3, 4]",
        "X = randn(0, [2, 3])",
        "Y = matmul(X, W)",
        "g = grad(mean(Y), W)",
        ":describe g",
        "A = randn(1, [2, 3])",
        "B = randn(2, [2, 3])",
        ":untag A",
        ":untag B",
        "B_lp = log_softmax(B, 1)",
        "L1 = randn(3, [2, 3])",
        "L2 = randn(4, [2, 3])",
        "sum_logits = L1 + L2",
        ":describe sum_logits",
        "mean_loss = mean(L1 + L2)",
        "L_tag = randn(5, [2, 3])",
        "P_tag = softmax(L_tag, 1)",
        "mix = L_tag + P_tag",
        ":untag P_tag",
        "mix = L_tag + P_tag",
        ":describe mix",
        "mdl = chain(linear(3, 4, 0), softmax_layer())",
        "out = apply(mdl, randn(0, [2, 3]))",
        ":describe out",
    ],
    try_it: "After the cross_entropy(probs, T) line failed with the double-softmax tutoring hint, try cross_entropy(softmax(L, 1), T) -- same bug, inline form. Then call :describe on every Weight in the workspace: walk through :vars first to find the auto-generated names like __linear_W_0. Finally, build a chain with a relu_layer tail and inspect apply(mdl, X) -- the result is tagged Activation(layer, kind=Relu), not Logit, because the structural-tail walk reaches relu_layer before any Linear.",
};

/// Saga 16 / 16.5 tutorial lesson.
pub const EMBEDDING_EXPLORATION: Lesson = Lesson {
    title: "Embedding exploration",
    intro: "Saga 16 + 16.5 ship five builtins for inspecting any rank-2 [N, D] array you want to treat as a set of points. pairwise_sqdist(X) returns the [N, N] squared-Euclidean distance matrix; knn(X, k) returns each row's k nearest non-self neighbors sorted by ascending distance; tsne(X, perplexity, iters, seed) runs classic van der Maaten t-SNE to reduce to [N, 2]; pca(X, k) returns the top-k PCA projection [N, k] via power iteration + Gram-Schmidt deflation (v0.14.1); embed_table(model) walks a ModelSpec tree and returns the first Embedding layer's [vocab, d_model] table (v0.14.1). Plus svg(pts, \"scatter3d\") renders [N, 3] as an orthographic 3-D scatter with axis gizmos. This lesson runs a 6-point fixture in 3-D so every render is instant in the browser; demos/embedding_viz.mlpl has the training story with a learned [12, 8] embedding table and docs/using-embeddings.md has the retrospective.",
    examples: &[
        "X = reshape([0.0, 0.0, 2.0, 0.1, 0.1, 2.0, -0.1, 0.0, 2.1, 2.0, 0.0, 0.0, 2.1, 0.1, 0.0, 1.9, -0.1, 0.0], [6, 3])",
        "pairwise_sqdist(X)",
        "knn(X, 2)",
        "svg(X, \"scatter3d\")",
        "emb_2d = tsne(X, 2.0, 100, 7)",
        "svg(emb_2d, \"scatter\")",
        "pca_2d = pca(X, 2)",
        "svg(pca_2d, \"scatter\")",
        "emb = embed(6, 3, 0)",
        "svg(embed_table(emb), \"scatter3d\")",
    ],
    try_it: "knn(X, 2) should list indices from X's own cluster -- rows 0/1/2 are near [0,0,2] and rows 3/4/5 are near [2,0,0]. pca_2d vs emb_2d: t-SNE rotates and flips between seeds and emphasizes local structure; PCA is deterministic and linear, so pca_2d keeps the two clusters on a single axis. embed_table(emb) returns the raw [6, 3] lookup table of a freshly-initialized embedding layer -- untrained, so the scatter is a tiny gaussian cloud. Run train ...adam over emb and re-call embed_table to see the learned rows.",
};

/// Multi-head attention built from primitives. Lives here
/// rather than inline in `lessons.rs` to keep that file
/// under its file-LOC budget.
pub const MULTI_HEAD_ATTENTION: Lesson = Lesson {
    title: "Multi-Head Attention from Scratch",
    intro: "Multi-head attention runs `h` copies of single-head attention in parallel on `d_k = d_model / h`-wide slabs, then concatenates the per-head outputs and projects through one final Wo. MLPL has no surface column-slicing op, so each head's slab is built explicitly via a selector matrix S_h: [d_model, d_k] (1s where this head's columns belong, 0s elsewhere). Multiplying full-width Q / K / V by S_h projects them down to the per-head width; multiplying the per-head output by S_h^T scatters it back into the full d_model width with zeros in the other heads' columns. Summing across heads recovers the concatenation. Each head's [T, T] weight heatmap shows a distinct attention pattern -- the model can dedicate one head per type of relationship in the input.",
    examples: &[
        "T = 4",
        "d_model = 4",
        "heads = 2",
        "d_k = d_model / heads",
        "X : [seq, d] = randn(0, [T, d_model])",
        "Wq = randn(1, [d_model, d_model])",
        "Wk = randn(2, [d_model, d_model])",
        "Wv = randn(3, [d_model, d_model])",
        "S0 = [[1,0],[0,1],[0,0],[0,0]]",
        "S1 = [[0,0],[0,0],[1,0],[0,1]]",
        "Q = matmul(X, Wq)",
        "K = matmul(X, Wk)",
        "V = matmul(X, Wv)",
        "W0 = softmax(matmul(matmul(Q, S0), transpose(matmul(K, S0))) / sqrt(d_k), 1)",
        "W1 = softmax(matmul(matmul(Q, S1), transpose(matmul(K, S1))) / sqrt(d_k), 1)",
        "svg(W0, \"heatmap\")",
        "svg(W1, \"heatmap\")",
        "out0 = matmul(W0, matmul(V, S0))",
        "out1 = matmul(W1, matmul(V, S1))",
        "out = matmul(out0, transpose(S0)) + matmul(out1, transpose(S1))",
        "Wo = randn(7, [d_model, d_model])",
        "shape(matmul(out, Wo))",
    ],
    try_it: "Replace the from-scratch pipeline with the model layer: `m = attention(4, 2, 0); apply(m, X)` and `attention_weights(m, X)` (which returns `[heads, T, T]` for multi-head). The math is the same; the layer just hides the slicing behind a single name.",
};

/// Cross-attention from primitives. Same `softmax(Q K^T /
/// sqrt(d_k)) V` formula as self-attention, but Q comes from
/// a target sequence and K / V come from a separate source.
pub const CROSS_ATTENTION: Lesson = Lesson {
    title: "Cross-Attention from Scratch",
    intro: "Cross-attention is what couples a transformer's decoder to its encoder: each target-side query attends to every source-side key, then mixes the source's V vectors into the target output. Mathematically it's identical to self-attention -- `softmax(Q K^T / sqrt(d_k)) V` -- but the inputs are split: Q is built from the target sequence (here T_tgt=4 random rows), while K and V are built from the source (T_src=6). The weight heatmap is `[T_tgt, T_src]` -- non-square -- which is the visual giveaway. Each row is one target query's distribution over source positions; each output row is the corresponding weighted average of V. Stack one cross-attention block after a causal-self-attention block and you have a transformer decoder layer; MLPL has no built-in cross-attention layer because the from-scratch pipeline is the same primitives you already know.",
    examples: &[
        "T_tgt = 4",
        "T_src = 6",
        "d_model = 4",
        "X_tgt : [seq, d] = randn(0, [T_tgt, d_model])",
        "X_src : [seq, d] = randn(1, [T_src, d_model])",
        "Wq = randn(2, [d_model, d_model])",
        "Wk = randn(3, [d_model, d_model])",
        "Wv = randn(4, [d_model, d_model])",
        "Q = matmul(X_tgt, Wq)",
        "K = matmul(X_src, Wk)",
        "V = matmul(X_src, Wv)",
        "weights = softmax(matmul(Q, transpose(K)) / sqrt(d_model), 1)",
        "shape(weights)",
        "svg(weights, \"heatmap\")",
        "out = matmul(weights, V)",
        "shape(out)",
    ],
    try_it: "Set X_src = X_tgt and re-run -- the weight heatmap collapses to T x T (square) and you have self-attention. Cross-attention's distinguishing feature is just the non-square shape that comes from Q and K/V having different row counts.",
};

/// Single transformer encoder block via the model DSL.
pub const ENCODER_BLOCK: Lesson = Lesson {
    title: "Encoder Block",
    intro: "One layer of a transformer encoder is just `chain(residual(pre_norm + self_attn), residual(pre_norm + ffn))`: two sub-blocks, each with a pre-normalization and a skip connection. The first sub-block lets every position mix information with every other (no causal mask in an encoder); the second sub-block applies a position-wise nonlinear transformation via two linear layers with a relu in between. Stacking N of these is what BERT does; replacing self-attention with causal self-attention and adding a third (cross-attention) sub-block is what a decoder layer does. The model DSL builds the whole block in five lines, and `attention_weights(encoder, X)` walks the chain to render the [T, T] attention pattern of just the self-attn sub-block.",
    examples: &[
        "X = randn(0, [4, 8])",
        "encoder = chain(residual(chain(rms_norm(8), attention(8, 1, 1))), residual(chain(rms_norm(8), linear(8, 16, 2), relu_layer(), linear(16, 8, 3))))",
        "out = apply(encoder, X)",
        "shape(out)",
        "bare_attn = attention(8, 1, 1)",
        "svg(attention_weights(bare_attn, X), \"heatmap\")",
    ],
    try_it: "Build a second encoder block inline (same chain expression with fresh seeds 4/5/6) and chain them: deeper = chain(encoder, encoder2). Shape is unchanged; the attention pattern at each layer differs because layer 2 sees layer 1's output. Real transformers stack 12-100+ blocks.",
};

/// Single transformer decoder block. Three sub-blocks:
/// causal self-attention, cross-attention (built from
/// scratch since MLPL has no cross-attn layer primitive),
/// and feedforward.
pub const DECODER_BLOCK: Lesson = Lesson {
    title: "Decoder Block",
    intro: "A transformer decoder block is the encoder block plus a cross-attention sub-block in the middle. Three sub-blocks total: (1) causal self-attention -- the target position can only attend to itself and earlier positions, enforced by `causal_attention(d, h, s)`'s lower-triangular mask; (2) cross-attention -- target queries (Q from `H`) attend to encoder output (K, V from `X_src`), built from scratch with matmul + softmax because MLPL has no cross-attention layer primitive; (3) feedforward -- the same `linear -> relu -> linear` MLP as the encoder. Each sub-block has its own pre-norm + residual. Stack a dozen of these and you have GPT (decoder-only, drop the cross-attn step) or T5 (encoder-decoder, keep all three). The cross-attention heatmap is `[T_tgt, T_src]` -- non-square -- which visually distinguishes it from self-attention.",
    examples: &[
        "T_tgt = 4",
        "T_src = 6",
        "d_model = 8",
        "d_ff = 16",
        "X_tgt = randn(0, [T_tgt, d_model])",
        "X_src = randn(1, [T_src, d_model])",
        "self_attn = residual(chain(rms_norm(d_model), causal_attention(d_model, 1, 2)))",
        "H = apply(self_attn, X_tgt)",
        "pre_xattn = rms_norm(d_model)",
        "H_norm = apply(pre_xattn, H)",
        "Wq = randn(3, [d_model, d_model])",
        "Wk = randn(4, [d_model, d_model])",
        "Wv = randn(5, [d_model, d_model])",
        "weights = softmax(matmul(matmul(H_norm, Wq), transpose(matmul(X_src, Wk))) / sqrt(d_model), 1)",
        "svg(weights, \"heatmap\")",
        "H2 = H + matmul(weights, matmul(X_src, Wv))",
        "ffn = residual(chain(rms_norm(d_model), linear(d_model, d_ff, 6), relu_layer(), linear(d_ff, d_model, 7)))",
        "out = apply(ffn, H2)",
        "shape(out)",
    ],
    try_it: "Drop the cross-attention sub-block (just leave self-attn + ffn) and you have a decoder-only block in the GPT style. Or stack the encoder demo's encoder before this decoder, feeding `apply(encoder, X_src)` as the new X_src -- that's the encoder-decoder pipeline, which Saga 24 (deferred) will package as a built-in.",
};

// --- Orientation triplet (course-outline Phase 1 gap fill).
// These are pure-prose lessons that frame the rest of the
// tutorial; they live in lessons_advanced.rs to keep
// lessons.rs under its file-LOC budget but are referenced
// at the TOP of `LESSONS` so they appear first in the UI.

/// "What is ML?" -- the destination-setting intro the
/// course outline calls out as missing.
pub const WHAT_IS_ML: Lesson = Lesson {
    title: "What is ML, and why are we here?",
    intro: "Machine learning is what you do when you cannot write the rules but you can show examples. Instead of `if x > 0.5 then ...`, you collect labeled data, define a parameterized function (a model), pick a loss that scores its predictions against the labels, and use gradient descent to nudge the parameters toward lower loss. Every classifier, regressor, MLP, attention block, and language model in this tutorial is a variation on that recipe -- linear regression is the smallest example, GPT is the largest. MLPL exists to make the variation visible: the inner loop of any model is a few lines of array code, not a wall of framework tensors. Read each lesson's `examples` block as a small worked example of the recipe; the `try_it` line at the end is a knob to twist.",
    examples: &[
        "X = [[0,0],[0,1],[1,0],[1,1]]",
        "y = [0, 0, 0, 1]",
        "w = zeros([2])",
        "b = 0",
        "z = matmul(X, reshape(w, [2, 1])) + b",
        "pred = sigmoid(z)",
        "loss = mean((pred - reshape(y, [4, 1])) * (pred - reshape(y, [4, 1])))",
        "loss",
    ],
    try_it: "The next lessons walk the recipe end-to-end on increasingly capable models. \"Hello Numbers\" starts with the smallest possible MLPL expressions; \"Logistic Regression\" trains the toy classifier this lesson sketched; \"Tiny LM\" trains a 1-layer transformer.",
};

/// "A short history of ML" -- Perceptron through
/// Transformers, with MLPL one-liners for the eras that
/// MLPL ships primitives for.
pub const HISTORY_OF_ML: Lesson = Lesson {
    title: "A short history of ML",
    intro: "Each ML era introduced one architectural idea that solved a problem the prior generation could not. The **Perceptron** (Rosenblatt 1958) -- a single linear layer plus a threshold -- proved that a machine could learn from labeled examples; AI winter set in when Minsky and Papert showed it could not solve XOR. The **MLP** (multi-layer perceptron) crossed that gap by adding a hidden layer + a non-linear activation, but training was unwieldy until **backpropagation** was popularized (Rumelhart, Hinton, Williams 1986). **CNNs** (LeCun 1989) used weight-sharing and convolutions to handle images; **RNNs** and **LSTMs** (1997) handled sequences. The **Transformer** (Vaswani et al. 2017) replaced recurrence with attention, unlocking the LM era we are in now. MLPL ships the Perceptron, MLP, and Transformer pieces directly; CNN and RNN are deferred so the historical arc is teachable but not yet runnable in MLPL.",
    examples: &[
        "perceptron = chain(linear(2, 1, 0), tanh_layer())",
        "shape(apply(perceptron, [[0,0],[0,1],[1,0],[1,1]]))",
        "mlp = chain(linear(2, 4, 1), tanh_layer(), linear(4, 1, 2))",
        "shape(apply(mlp, [[0,0],[0,1],[1,0],[1,1]]))",
        "attn_layer = attention(8, 1, 3)",
        "shape(apply(attn_layer, randn(0, [4, 8])))",
    ],
    try_it: "The progression from perceptron to MLP to transformer is two architectural leaps: adding a hidden layer (Perceptron -> MLP solves XOR) and replacing matrix mixing across positions with attention (MLP -> Transformer solves long-range dependencies). Try classifying the XOR data with `perceptron` (it cannot) and then with `mlp` (it can) by training each through `train` blocks.",
};

/// "How models learn" -- companion to HISTORY_OF_ML, but
/// walking training paradigms (supervised / unsupervised
/// / self-supervised / RLHF / distillation / self-play)
/// instead of architectures.
pub const HOW_MODELS_LEARN: Lesson = Lesson {
    title: "How models learn: a short history of training paradigms",
    intro: "Architecture (Perceptron / MLP / Transformer) is half the story of ML history; the other half is how models get their training signal. **Supervised learning** -- the classical paradigm -- minimizes a per-example loss against human-supplied labels. The Logistic Regression / Tiny MLP / Softmax Classifier demos are all supervised. **Unsupervised learning** drops the labels entirely and discovers structure from geometry alone -- K-Means clusters, PCA finds the principal axis, t-SNE preserves local distances. **Self-supervised learning** keeps the supervised loss but derives the labels from the input itself; next-token prediction is the canonical example, and MLPL's `shift_pairs_x` / `shift_pairs_y` pairing in the Tiny LM demo is exactly this -- the label for token t is just token t+1, no humans involved. The LLM era is mostly self-supervised pretraining followed by smaller human-touched stages: **RLHF** (preference learning over (chosen, rejected) pairs, deferred in MLPL), **distillation** (train a student against a teacher's softened logits via KL-divergence; deferred -- needs `kl_div` / `soft_targets` builtins), and **self-play** (an agent generates its own training signal by playing itself, the AlphaGo / AlphaZero pattern; deferred -- needs environment + reward primitives). The arc isn't strictly chronological -- supervised learning never went away, it just stopped being enough on its own at scale.",
    examples: &[
        "y_true = [0, 1, 0, 1]                                      # supervised: classical paradigm",
        "logits = [[2.0, -1.0], [-0.5, 1.5], [1.0, 0.0], [-1.0, 1.0]]",
        "supervised_loss = cross_entropy(logits, y_true)            # human-supplied labels minimize CE",
        "supervised_loss",
        "X = randn(0, [60, 2])                                      # unsupervised: structure from geometry",
        "X_pca = pca(matmul(X, [[1, 2], [0, 0.3]]), 2)              # PCA finds the long axis -- no labels",
        "shape(X_pca)",
        "ids = [3, 7, 2, 9, 4, 1, 8, 5, 6, 0]                       # self-supervised: derive labels from data",
        "self_X = shift_pairs_x(ids, 4)                             # input windows of length 4",
        "self_Y = shift_pairs_y(ids, 4)                             # label = input shifted right by one",
        "shape(self_X)",
        "shape(self_Y)",
    ],
    try_it: "The three working examples cover supervised / unsupervised / self-supervised. RLHF, distillation, and self-play are deferred until MLPL ships preference loss + `kl_div` + environment primitives. Try sketching distillation by hand: pick a teacher logit vector `t = [2.0, -1.0, 0.5]`, soften with temperature 2 via `softmax(t / 2.0, 0)`, and a student via `softmax(s / 2.0, 0)`; the KL loss is `reduce_add(p * (log(p) - log(q)))` -- that's the formula a future `kl_div` builtin would wrap. The Module 11 distillation gap in `docs/course-outline.md` traces exactly this missing piece.",
};

/// "Why backprop?" -- the historical complement to
/// "Automatic Differentiation". Frames `grad` as the
/// generalization of hand-derived chain-rule formulas.
pub const WHY_BACKPROP: Lesson = Lesson {
    title: "Why backprop?",
    intro: "Backpropagation is just the chain rule applied to a computation graph. Before it was popularized in 1986, the gradient of a model's loss was a per-architecture derivation: hand-derive `dW = X^T (pred - y) / N` for linear regression, a different formula for logistic regression, and a third for a two-layer MLP. Backprop generalizes that work: build the forward graph; the backward walk produces every gradient automatically. The \"Logistic Regression\" and \"Tiny MLP\" lessons in this tutorial show the manual chain-rule version (`dZ2 = pred - y`, `dZ1 = (dZ2 W2^T) * (1 - H * H)`); the \"Automatic Differentiation\" lesson shows the `grad(loss, wrt)` version that automates it. Both compute the same gradient. The win is not faster math -- it is faster *iteration*: changing your loss or your model no longer means redoing pages of derivative calculus.",
    examples: &[
        "X = [[1.0], [2.0], [3.0]]",
        "y = [2.0, 4.0, 6.0]",
        "W = param[1, 1]",
        "W = randn(0, [1, 1])",
        "pred = matmul(X, W)",
        "manual_dW = matmul(transpose(X), reshape(pred - reshape(y, [3, 1]), [3, 1])) / 3.0",
        "loss = mean((pred - reshape(y, [3, 1])) * (pred - reshape(y, [3, 1])))",
        "auto_dW = grad(loss, \"W\")",
        "manual_dW",
        "auto_dW",
    ],
    try_it: "Confirm `manual_dW` and `auto_dW` are the same up to floating-point noise. Then change the loss expression (e.g. add a regularizer `+ 0.01 * reduce_add(W * W)`) and watch how `grad` keeps working without any hand-derivation -- which is the entire point of backprop.",
};
