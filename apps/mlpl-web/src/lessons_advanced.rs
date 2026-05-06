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
        "T = 4  # Sequence length. Small so each [T, T] head heatmap renders large enough to read.",
        "d_model = 4  # Total embedding width PER TOKEN. Will be split across heads.",
        "heads = 2  # Number of parallel attention heads. Real transformers use 8-32; we use 2 so you can see the per-head difference.",
        "d_k = d_model / heads  # Per-head width. With d_model=4 and heads=2, each head sees 2 dims of every token. Smaller heads = more specialization, more total parameters in Wq/Wk/Wv stays constant.",
        "X : [seq, d] = randn(0, [T, d_model])  # Input: [4, 4] random rows. Each row is one token's full d_model-wide embedding.",
        "Wq = randn(1, [d_model, d_model])  # FULL-WIDTH query projection. Learns one combined mapping that gets sliced into per-head views below. Real transformers often skip the slicing and just have h smaller projection matrices, but the math is identical.",
        "Wk = randn(2, [d_model, d_model])  # Full-width key projection. Same shape as Wq.",
        "Wv = randn(3, [d_model, d_model])  # Full-width value projection. Same shape.",
        "S0 = [[1,0],[0,1],[0,0],[0,0]]  # Selector for head 0: a [d_model, d_k] matrix that picks COLUMNS 0 and 1 of any tensor it left-multiplies. Crude column-slicing, since MLPL has no surface slice op.",
        "S1 = [[0,0],[0,0],[1,0],[0,1]]  # Selector for head 1: picks columns 2 and 3. Together S0 and S1 partition d_model into the two heads.",
        "Q = matmul(X, Wq)  # Project everything once, then split per-head below. Keeps the matmul count down and matches what real implementations do.",
        "K = matmul(X, Wk)  # Same for keys.",
        "V = matmul(X, Wv)  # Same for values.",
        "W0 = softmax(matmul(matmul(Q, S0), transpose(matmul(K, S0))) / sqrt(d_k), 1)  # Head 0's [T, T] attention weights. Q*S0 = head-0 slab of Q ([T, d_k]); K*S0 = head-0 slab of K. Then standard scaled-dot-product softmax. Note sqrt(d_k), NOT sqrt(d_model) -- each head normalizes by ITS dimension.",
        "W1 = softmax(matmul(matmul(Q, S1), transpose(matmul(K, S1))) / sqrt(d_k), 1)  # Head 1's [T, T] attention weights. Same expression, different selector.",
        "svg(W0, \"heatmap\")  # Render head 0's attention pattern.",
        "svg(W1, \"heatmap\")  # Render head 1's attention pattern. Compare it to W0 -- the two heads produce DIFFERENT patterns even though they read the same input X. That is the entire point of multi-head: each head can specialize.",
        "out0 = matmul(W0, matmul(V, S0))  # Mix head 0's weights with head 0's V slab. Output is [T, d_k] for this head.",
        "out1 = matmul(W1, matmul(V, S1))  # Same for head 1.",
        "out = matmul(out0, transpose(S0)) + matmul(out1, transpose(S1))  # Scatter each head's narrow [T, d_k] output back into the full [T, d_model] width via S_h^T (zeros in the OTHER head's columns), then sum. This recovers what real implementations do via concatenate.",
        "Wo = randn(7, [d_model, d_model])  # Output projection. After concat, the heads' outputs go through one final mixing matrix Wo. This is what lets information FLOW between the heads' specializations.",
        "shape(matmul(out, Wo))  # [T, d_model] -- back to input shape. Multi-head attention is also shape-preserving.",
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
        "T_tgt = 4  # Target sequence length (4 rows of TARGET tokens). In a translation pipeline, this is the partial English output the decoder is generating.",
        "T_src = 6  # Source sequence length (6 rows of SOURCE tokens). In translation, this is the encoded French input the decoder needs to attend to.",
        "d_model = 4  # Shared embedding width across both sequences. Cross-attention requires source and target to be in the same vector space.",
        "X_tgt : [seq, d] = randn(0, [T_tgt, d_model])  # Target embeddings. In a real model these come from the decoder's preceding self-attention layer.",
        "X_src : [seq, d] = randn(1, [T_src, d_model])  # Source embeddings. In a real model these come from the encoder's final layer (a [T_src, d_model] block of contextualized representations).",
        "Wq = randn(2, [d_model, d_model])  # Query projection. The TARGET side asks the questions, so Q is built from X_tgt below.",
        "Wk = randn(3, [d_model, d_model])  # Key projection. The SOURCE side advertises content, so K is built from X_src.",
        "Wv = randn(4, [d_model, d_model])  # Value projection. Also from the source -- the source provides the content that gets mixed into the target output.",
        "Q = matmul(X_tgt, Wq)  # Each target token emits its query. Shape [T_tgt, d_model].",
        "K = matmul(X_src, Wk)  # Each SOURCE token emits its key. Shape [T_src, d_model] -- different row count from Q.",
        "V = matmul(X_src, Wv)  # Each source token emits its value. Shape [T_src, d_model].",
        "weights = softmax(matmul(Q, transpose(K)) / sqrt(d_model), 1)  # The score matrix is [T_tgt, T_src] -- non-square. Each row is one target query's distribution over source keys. Per-row softmax = each target token's attention spend across the source.",
        "shape(weights)  # Confirm [T_tgt, T_src] = [4, 6]. Non-square is the cross-attention signature; if this came back square you would know you accidentally did self-attention.",
        "svg(weights, \"heatmap\")  # Render the cross-attention pattern. The 4-row, 6-column heatmap reveals which source positions each target token is reading from. In a trained translator, this often aligns roughly word-for-word.",
        "out = matmul(weights, V)  # Mix the source values according to the target-side weights. Each output row is one target token's weighted average of source content.",
        "shape(out)  # [T_tgt, d_model] = [4, 4]. The output has the TARGET's row count and the shared d_model width -- shape suitable for the decoder's next sub-block (FFN).",
    ],
    try_it: "Set X_src = X_tgt and re-run -- the weight heatmap collapses to T x T (square) and you have self-attention. Cross-attention's distinguishing feature is just the non-square shape that comes from Q and K/V having different row counts.",
};

/// Single transformer encoder block via the model DSL.
pub const ENCODER_BLOCK: Lesson = Lesson {
    title: "Encoder Block",
    intro: "One layer of a transformer encoder is just `chain(residual(pre_norm + self_attn), residual(pre_norm + ffn))`: two sub-blocks, each with a pre-normalization and a skip connection. The first sub-block lets every position mix information with every other (no causal mask in an encoder); the second sub-block applies a position-wise nonlinear transformation via two linear layers with a relu in between. Stacking N of these is what BERT does; replacing self-attention with causal self-attention and adding a third (cross-attention) sub-block is what a decoder layer does. The model DSL builds the whole block in five lines, and `attention_weights(encoder, X)` walks the chain to render the [T, T] attention pattern of just the self-attn sub-block.",
    examples: &[
        "X = randn(0, [4, 8])  # Input: 4 tokens, d_model=8 each. We are about to push X through one full encoder block and check that shape comes out unchanged.",
        "encoder = chain(residual(chain(rms_norm(8), attention(8, 1, 1))), residual(chain(rms_norm(8), linear(8, 16, 2), relu_layer(), linear(16, 8, 3))))  # The whole block in one expression. Outer `chain` runs sub-block 1 then sub-block 2. Each sub-block is `residual(chain(pre_norm, layer))` -- so the actual computation is X + layer(rms_norm(X)). Sub-block 1's layer is single-head self-attention. Sub-block 2's layer is FFN: linear(8 -> 16) -> relu -> linear(16 -> 8).",
        "out = apply(encoder, X)  # One forward pass through both sub-blocks. The model layer hides all the internal apply() calls -- you just give it X.",
        "shape(out)  # [4, 8] -- same as the input. Shape preservation is what makes encoder blocks STACKABLE: out becomes the input of the next block.",
        "bare_attn = attention(8, 1, 1)  # A standalone self-attention layer with the SAME seed as the one buried inside `encoder`. This is the only way to extract attention_weights -- the model walker does not recurse into nested chains/residuals.",
        "svg(attention_weights(bare_attn, X), \"heatmap\")  # Render the [T, T] attention pattern of the bare self-attention layer applied to X (NOT to the post-rms_norm transformed input -- this is an approximation). With random weights the pattern is noise; train the encoder on a real task and structure emerges.",
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
        "T_tgt = 4  # Target sequence length. In language generation, this is the number of tokens the decoder has produced so far.",
        "T_src = 6  # Source sequence length. Stand-in for the encoder's output -- in a real pipeline, replace X_src below with `apply(encoder, X)`.",
        "d_model = 8  # Shared embedding width. Both the target and source must live in the same d_model space for cross-attention to work.",
        "d_ff = 16  # FFN hidden width. Conventional ratio is d_ff = 4 * d_model; we use 2x to keep numbers small.",
        "X_tgt = randn(0, [T_tgt, d_model])  # Target tokens. In generation these come from the previous decoder layer; here they are random.",
        "X_src = randn(1, [T_src, d_model])  # Source tokens (encoder output stand-in).",
        "self_attn = residual(chain(rms_norm(d_model), causal_attention(d_model, 1, 2)))  # Sub-block 1: causal self-attention with pre-norm + residual. The CAUSAL mask is what makes a decoder a decoder -- token t can only attend to itself and earlier targets, never to future ones. (At training time the whole sequence is visible; the mask makes the model behave as if it weren't.)",
        "H = apply(self_attn, X_tgt)  # Run sub-block 1. H has the same shape as X_tgt; each row now mixes information from earlier target rows.",
        "pre_xattn = rms_norm(d_model)  # Pre-normalization for the cross-attention sub-block. We apply it to H (the target side) before computing queries.",
        "H_norm = apply(pre_xattn, H)  # Normalized H, ready to be the source of cross-attention queries.",
        "Wq = randn(3, [d_model, d_model])  # Cross-attention Q projection (target side).",
        "Wk = randn(4, [d_model, d_model])  # Cross-attention K projection (source side).",
        "Wv = randn(5, [d_model, d_model])  # Cross-attention V projection (source side).",
        "weights = softmax(matmul(matmul(H_norm, Wq), transpose(matmul(X_src, Wk))) / sqrt(d_model), 1)  # The cross-attention score matrix is [T_tgt, T_src] -- non-square -- with per-row softmax. Each target token now has a probability distribution over source positions saying 'which source token am I copying from?'",
        "svg(weights, \"heatmap\")  # Render the [T_tgt, T_src] heatmap. In a trained encoder-decoder this often shows a roughly diagonal alignment -- target token i tends to attend to source token i (with phrase-level reordering for tasks like translation).",
        "H2 = H + matmul(weights, matmul(X_src, Wv))  # The cross-attention residual: H + (cross_attn output). matmul(X_src, Wv) is V; weights @ V mixes source values per target query. Add to H to keep the residual signal.",
        "ffn = residual(chain(rms_norm(d_model), linear(d_model, d_ff, 6), relu_layer(), linear(d_ff, d_model, 7)))  # Sub-block 3: position-wise feedforward, same as encoder. Pre-norm + linear -> relu -> linear + residual.",
        "out = apply(ffn, H2)  # Final decoder block output. Same shape as X_tgt: stackable.",
        "shape(out)  # [T_tgt, d_model] = [4, 8]. Decoder blocks preserve target shape; stack a dozen and you have a real decoder. Drop the cross-attention step (sub-block 2) and you have a GPT-style decoder-only block.",
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
    intro: "Architecture (Perceptron / MLP / Transformer) is half the story of ML history; the other half is **how models get their training signal**.\n\n## Working in MLPL today\n\n- **Supervised learning** -- the classical paradigm. Minimize a per-example loss against human-supplied labels (Logistic Regression, Tiny MLP, Softmax Classifier demos).\n- **Unsupervised learning** -- drop the labels entirely and discover structure from geometry alone. K-Means clusters, PCA finds the principal axis, t-SNE preserves local distances.\n- **Self-supervised learning** -- keep the supervised loss, but derive the labels from the input itself. Next-token prediction is the canonical example: MLPL's `shift_pairs_x` / `shift_pairs_y` pairing in the Tiny LM demo is exactly this. The label for token t is just token t+1, no humans involved.\n\n---\n\n## The LLM-era stack (mostly deferred in MLPL today)\n\nModern LLMs start with a long self-supervised pretraining phase, then layer on smaller human-touched stages:\n\n- **RLHF** -- preference learning over (chosen, rejected) pairs. Deferred: needs a preference loss surface.\n- **Distillation** -- train a student against a teacher's softened logits via KL-divergence. Deferred: needs `kl_div` / `soft_targets` builtins (the Module 11 blocker in `docs/course-outline.md`).\n- **Self-play** -- an agent generates its own training signal by playing itself, the AlphaGo / AlphaZero pattern. Deferred: needs environment + reward primitives.\n\n---\n\n## Why the arc isn't strictly chronological\n\nSupervised learning never went away -- it just stopped being enough on its own at scale. Today's frontier models still use it, layered between self-supervised pretraining and the human-touched stages above. Each new paradigm extends the toolkit; none replace what came before.",
    examples: &[
        "y_true = [0, 1, 0, 1]  # Supervised paradigm: each input row needs a known label, supplied by a human or curation pipeline. Here y_true is a length-4 vector of class ids -- we are TELLING the model the right answer for each example.",
        "logits = [[2.0, -1.0], [-0.5, 1.5], [1.0, 0.0], [-1.0, 1.0]]  # A 4-row x 2-class logit matrix. In a real classifier this would be the output of `apply(model, X)`; here we hardcode it so the loss math is visible without training.",
        "supervised_loss = cross_entropy(logits, y_true)  # cross_entropy(logits, labels) is the canonical supervised loss for classification. Minimizing it via gradient descent makes the model assign higher logit to the correct class on each row.",
        "supervised_loss  # The scalar loss prints. Lower = better. In a real training loop we would call grad(supervised_loss, ...) and step the weights.",
        "X = randn(0, [60, 2])  # Unsupervised paradigm: no labels at all. We start with 60 random points in 2D and ask: what STRUCTURE is in this data? The next line manufactures structure by stretching the cloud along one axis.",
        "X_pca = pca(matmul(X, [[1, 2], [0, 0.3]]), 2)  # Stretch X anisotropically with a 2x2 transform, then project onto the top 2 principal components. PCA found the high-variance axis using only geometry -- zero labels were used.",
        "shape(X_pca)  # [60, 2] -- same shape as the (stretched) input. PCA is a CHANGE OF BASIS to the axes ranked by variance, not a dimensionality cut here.",
        "ids = [3, 7, 2, 9, 4, 1, 8, 5, 6, 0]  # Self-supervised paradigm: a sequence of token ids. Pretend this came from running tokenize_bytes on a corpus. There are no human labels, but we will MANUFACTURE labels from the data itself on the next line.",
        "self_X = shift_pairs_x(ids, 4)  # Build input windows of length 4 by sliding over `ids`. Each window says: 'given these 4 tokens...'",
        "self_Y = shift_pairs_y(ids, 4)  # ...what is the very next token? The label for window i is just ids[i+4]. No human ever said 'this is the right answer' -- we DERIVED labels from the input itself. This is exactly how every modern LLM is pretrained.",
        "shape(self_X)  # Confirm the [windows, 4] shape -- each row is an X for one training example.",
        "shape(self_Y)  # Same number of rows as self_X. The (X, Y) pair feeds straight into a supervised loss like cross_entropy(apply(model, X), Y), but no human labels exist.",
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
