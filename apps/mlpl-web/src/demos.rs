pub struct Demo {
    pub name: &'static str,
    /// One-to-three-sentence framing shown before the demo runs:
    /// what the demo does and why. Intentionally short -- the
    /// code is the real lesson.
    pub intro: &'static str,
    /// One-to-three-sentence takeaway shown after the demo's last
    /// line completes: what the output proves and where to go
    /// next. Paired with `intro` to bookend the run.
    pub takeaway: &'static str,
    pub lines: &'static [&'static str],
}

/// A heads-up rendered before a single long-running demo line.
/// Browser WASM evaluates each line on the main thread, so a
/// 30-step train block (Tiny LM) blocks the event loop for
/// seconds. Without a note the user sees a previous line's
/// output, then a stalled tab, then the result. The note paints
/// before the line starts so the wait is intentional and
/// estimated, not mysterious.
#[derive(Clone, Copy)]
pub struct ProgressNote {
    /// Demo's `name` field.
    pub demo: &'static str,
    /// Index into `Demo::lines` that the note precedes.
    pub line_idx: usize,
    /// Short heading -- e.g. "Training the language model".
    pub heading: &'static str,
    /// One-to-three-sentence body explaining what the runtime
    /// is about to do and a rough ETA on a recent laptop.
    pub body: &'static str,
}

/// Heads-up notes for demos whose individual lines block the
/// event loop long enough that the user wonders if the page is
/// frozen. Each entry attaches to a specific demo + line index.
/// Demos not listed here render with no pre-line narration --
/// the existing intro / takeaway pair is enough.
///
/// ETA wording is approximate (modern laptop, no MLX). The
/// numbers err on the high side so a faster machine never
/// sees the heads-up linger longer than the actual op.
pub const PROGRESS_NOTES: &[ProgressNote] = &[
    ProgressNote {
        demo: "Tiny LM Generate",
        line_idx: 1,
        heading: "Training the BPE tokenizer",
        body: "train_bpe walks the corpus and learns 260 byte-pair merges. A few seconds; the corpus is small but the merge loop is O(merges * pairs).",
    },
    ProgressNote {
        demo: "Tiny LM Generate",
        line_idx: 9,
        heading: "Training the language model (~10-30s)",
        body: "30 Adam steps over a 1-layer transformer (V=260, d=16, block=8). Each step runs a forward pass, a backward pass through the autograd tape, and an Adam update over every model parameter. The browser tab is single-threaded WASM so the page is unresponsive for the duration; this is normal.",
    },
    ProgressNote {
        demo: "Tiny LM Generate",
        line_idx: 13,
        heading: "Generating 20 tokens (~3-8s)",
        body: "Each token is a forward pass on the growing sequence, top_k restriction, and a multinomial sample. The output is decoded BPE bytes back to text once the loop finishes.",
    },
    ProgressNote {
        demo: "Tiny LM",
        line_idx: 1,
        heading: "Training the BPE tokenizer",
        body: "train_bpe walks the corpus and learns 260 byte-pair merges. A few seconds.",
    },
    ProgressNote {
        demo: "Tiny LM",
        line_idx: 9,
        heading: "Training the language model (~10-30s)",
        body: "30 Adam steps over a 1-layer transformer (V=260, d=16, block=8). The page is unresponsive while WASM runs on the main thread; the loss curve renders once the train block returns.",
    },
    ProgressNote {
        demo: "Tiny MLP",
        line_idx: 10,
        heading: "Training the hand-rolled MLP (~5-10s)",
        body: "600 iterations of an explicit forward + backward pass on 80 points. No autograd -- every gradient is written out so you can see the chain rule run.",
    },
    ProgressNote {
        demo: "Moons MLP",
        line_idx: 7,
        heading: "Training with Adam (~10-20s)",
        body: "200 train steps on 120 points through the autograd tape. The decision-boundary surface that follows is the visible payoff.",
    },
    ProgressNote {
        demo: "Logistic Regression",
        line_idx: 7,
        heading: "Gradient descent (~2-5s)",
        body: "300 iterations of explicit logistic-regression gradient descent on four points. Short but visibly blocks the tab.",
    },
    ProgressNote {
        demo: "Softmax Classifier",
        line_idx: 9,
        heading: "Training the softmax classifier (~3-8s)",
        body: "300 explicit-gradient steps over 90 points and 3 classes. The decision-boundary plot at the end shows the three wedges this minimization carved out.",
    },
    ProgressNote {
        demo: "Pets: cat vs dog (quick)",
        line_idx: 33,
        heading: "Training the Vision Transformer (~30-60s)",
        body: "30 full-batch Adam steps on the 8 pet images you just assembled. Each step runs the full forward pipeline (patchify -> linear-embed -> rank-3 attention -> first-token pooling -> 2-layer MLP classifier), pushes a tape of ~hundreds of nodes, runs the backward pass to compute gradients on every model parameter, then applies one Adam update. WASM is single-threaded so the page is unresponsive for the duration; this is normal, not a hang. The loss curve renders right after the train block returns -- you'll see cross-entropy drop from ~0.69 (random) toward 0 (perfect overfit on 8 images).",
    },
    ProgressNote {
        demo: "Pets: predict + gallery",
        line_idx: 61,
        heading: "Training the Vision Transformer (6 chunks x 5 steps, ~30-60s)",
        body: "30 full-batch Adam steps on 16 images, run as 6 separate train blocks of 5 steps each. Each chunk runs synchronously in WASM (~5s of unresponsive UI); the demo runner yields to the browser between chunks so the tab stays responsive end-to-end and you can see incremental progress. Adam momentum persists across chunks via the session env, so this is mathematically equivalent to one `train 30`. After training, the model predicts labels for all 16 images and renders a labeled gallery (actual / predicted under each thumbnail). 0 = cat, 1 = dog.",
    },
    ProgressNote {
        demo: "Pets: predict + gallery",
        line_idx: 73,
        heading: "Rendering the loss curve + labeled gallery",
        body: "Training is done. Below: the 30-step cross-entropy loss curve (concatenated across the six train chunks), then `predict_batch` runs the trained model over all 16 images, then `svg(X, \"gallery\", preds_2col)` renders the labeled thumbnail grid. Misclassifications stand out because the two captions under a thumbnail disagree (0/1 or 1/0).",
    },
];

/// Look up progress notes for `(demo_name, line_idx)`. Returns
/// the matching slice (usually 0 or 1 entries) without
/// allocating; callers iterate.
pub fn progress_notes_for(
    demo_name: &str,
    line_idx: usize,
) -> impl Iterator<Item = &'static ProgressNote> {
    PROGRESS_NOTES
        .iter()
        .filter(move |n| n.demo == demo_name && n.line_idx == line_idx)
}

// Demos are listed alphabetically by `name`.
pub const DEMOS: &[Demo] = &[
    Demo {
        name: "Analysis Helpers",
        intro: "Tour of the high-level viz helpers: histogram, labeled scatter, loss curve, confusion matrix, and a 2D decision-boundary surface. Each returns an SVG that renders inline. Use these as building blocks when you want one line of code to answer one question about your data or model.",
        takeaway: "Six one-liners, six labeled plots. Every helper under 'analysis' in the docs takes arrays you already have and returns an SVG -- no separate plotting library, no config blocks.",
        lines: &[
            "hist([1, 2, 2, 3, 3, 3, 4, 4, 5], 5)                                # histogram with 5 equal-width bins",
            "scatter_labeled([[0,0],[1,1],[0,1],[1,0]], [0, 0, 1, 1])            # 2D points colored by per-row label",
            "loss_curve([5, 3, 2, 1.5, 1.0, 0.7, 0.5, 0.4, 0.3, 0.25])           # training-loss line plot",
            "confusion_matrix([0, 1, 2, 1, 0], [0, 1, 1, 1, 0])                  # KxK predicted-vs-actual heatmap",
            "gx = grid([0, 1, 0, 1], 20)                                         # 20x20 (x, y) input-space grid",
            "boundary_2d(reshape(iota(400), [400]) / 400, [20, 20], [[0,0],[1,1]], [0, 1])  # synthetic gradient surface as boundary",
        ],
    },
    Demo {
        name: "Attention Pattern",
        intro: "The scaled-dot-product attention formula from 'Attention is All You Need', spelled out. Two Q and K matrices, scored and softmax'd, rendered as heatmaps. A second self-attention pass (Q attending to itself) shows the diagonal pattern that makes self-attention recognizable on sight.",
        takeaway: "You just built transformer attention in four lines: matmul + transpose + softmax. The heatmaps show where each query row puts probability mass -- self-attention concentrates on the diagonal; cross-attention doesn't.",
        lines: &[
            "Q = randn(17, [6, 4])                              # 6 queries, 4 dims each",
            "K = randn(23, [6, 4])                              # 6 keys (different seed = cross-attention)",
            "scores = matmul(Q, transpose(K)) / sqrt(4)         # scaled dot-product",
            "A = softmax(scores, 1)                             # row-wise probability over keys",
            "svg(scores, \"heatmap\")                            # raw scores (signed values)",
            "svg(A, \"heatmap\")                                 # attention weights (rows sum to 1)",
            "Qs = randn(17, [6, 4])                             # reuse the same matrix as both Q and K ...",
            "Sself = matmul(Qs, transpose(Qs)) / sqrt(4)        # ... and you have self-attention",
            "Aself = softmax(Sself, 1)                          # row-wise softmax",
            "svg(Aself, \"heatmap\")                             # diagonal-dominant: the self-attention signature",
        ],
    },
    Demo {
        name: "Self-Attention from Scratch",
        intro: "Build one head of self-attention from primitives, no `attention()` layer involved. Three projection matrices Wq / Wk / Wv turn the input X into Q, K, V; then `softmax(Q K^T / sqrt(d_k)) V` mixes the values according to which keys each query attended to. Tiny T=6, d=4 toy input so every intermediate matrix renders cleanly.",
        takeaway: "Self-attention is four matmuls, one transpose, one scalar division, one softmax. The score heatmap is what a query is asking; the weight heatmap is what it gets back; the output is the weighted V. With random projections the pattern is noise -- training is what makes the weights specialize on a task.",
        lines: &[
            "T = 6                                                # 6 tokens in the sequence",
            "d_model = 4                                          # each token is a 4-dim vector",
            "X : [seq, d] = randn(0, [T, d_model])                # the input: a [6, 4] block of random embeddings",
            "Wq = randn(1, [d_model, d_model])                    # query projection (learned in a real model)",
            "Wk = randn(2, [d_model, d_model])                    # key projection",
            "Wv = randn(3, [d_model, d_model])                    # value projection",
            "Q = matmul(X, Wq)                                    # what each token is asking about",
            "K = matmul(X, Wk)                                    # what each token advertises about itself",
            "V = matmul(X, Wv)                                    # what each token shares if attended-to",
            "scores = matmul(Q, transpose(K)) / sqrt(d_model)     # raw query-key similarity, scaled so softmax stays stable",
            "svg(scores, \"heatmap\")                               # visualize the unnormalized scores",
            "weights = softmax(scores, 1)                         # each row becomes a probability distribution over keys",
            "svg(weights, \"heatmap\")                              # visualize the attention pattern",
            "out = matmul(weights, V)                             # weighted average of value vectors per query",
            "shape(out)                                           # same shape as the input: [T, d_model]",
        ],
    },
    Demo {
        name: "Multi-Head Attention from Scratch",
        intro: "Multi-head attention from primitives. Each head gets a slab of the d_model channels via a column-selector matrix S_h: [d_model, d_k], runs scaled-dot-product attention on its own Q/K/V slabs, and writes back into the full-width output via S_h^T. Tiny T=4, d_model=4, heads=2 (so d_k=2) keeps every per-head [T, T] heatmap interpretable.",
        takeaway: "Two heads, two different attention patterns. Each head sees only its slab of d_model, so it can specialize. Real `attention(d, heads, seed)` does the slicing in Rust without materializing selector matrices, and adds a final Wo projection -- but the math is the same: per-head softmax(Q K^T / sqrt(d_k)) V, concat, project.",
        lines: &[
            "T = 4                                                # 4 tokens",
            "d_model = 4                                          # 4-dim vectors total ...",
            "heads = 2                                            # ... split into 2 heads ...",
            "d_k = d_model / heads                                # ... each head is 2 dims wide",
            "X : [seq, d] = randn(0, [T, d_model])                # the input: [T, d_model]",
            "Wq = randn(1, [d_model, d_model])                    # full-width Q projection",
            "Wk = randn(2, [d_model, d_model])                    # full-width K projection",
            "Wv = randn(3, [d_model, d_model])                    # full-width V projection",
            "S0 = [[1,0],[0,1],[0,0],[0,0]]                       # selector for head 0: picks columns 0..d_k",
            "S1 = [[0,0],[0,0],[1,0],[0,1]]                       # selector for head 1: picks columns d_k..d_model",
            "Q = matmul(X, Wq)                                    # full-width Q for all heads",
            "K = matmul(X, Wk)                                    # full-width K",
            "V = matmul(X, Wv)                                    # full-width V",
            "Q0 = matmul(Q, S0)                                   # head-0 slab of Q (T, d_k)",
            "K0 = matmul(K, S0)                                   # head-0 slab of K",
            "V0 = matmul(V, S0)                                   # head-0 slab of V",
            "W0 = softmax(matmul(Q0, transpose(K0)) / sqrt(d_k), 1) # head-0 attention weights",
            "svg(W0, \"heatmap\")                                   # head-0 pattern",
            "out0 = matmul(W0, V0)                                # head-0 output slab",
            "Q1 = matmul(Q, S1)                                   # head-1 slab of Q",
            "K1 = matmul(K, S1)                                   # head-1 slab of K",
            "V1 = matmul(V, S1)                                   # head-1 slab of V",
            "W1 = softmax(matmul(Q1, transpose(K1)) / sqrt(d_k), 1) # head-1 attention weights",
            "svg(W1, \"heatmap\")                                   # head-1 pattern (different from head-0)",
            "out1 = matmul(W1, V1)                                # head-1 output slab",
            "out_concat = matmul(out0, transpose(S0)) + matmul(out1, transpose(S1)) # scatter slabs back into [T, d_model] and sum",
            "Wo = randn(7, [d_model, d_model])                    # final output projection (mixes across heads)",
            "final_out = matmul(out_concat, Wo)                   # the multi-head output",
            "shape(final_out)                                     # back to [T, d_model]",
        ],
    },
    Demo {
        name: "Cross-Attention from Scratch",
        intro: "Cross-attention is the same `softmax(Q K^T / sqrt(d_k)) V` as self-attention, but Q comes from a target sequence (T_tgt=4) and K, V come from a separate source sequence (T_src=6). The weight matrix is `[T_tgt, T_src]` -- non-square -- which is the visual signature: each target query distributes its attention across all source positions, not over its own sequence.",
        takeaway: "The non-square weight heatmap (4 rows, 6 cols) is the giveaway: target queries are looking at source content, not at themselves. The output is `[T_tgt, d_model]` -- same shape as the target input -- but its values are mixtures of the source's V vectors weighted by query-key similarity. Stack one cross-attention block after a causal-self-attention block and you have a transformer decoder layer.",
        lines: &[
            "T_tgt = 4                                            # target sequence length (decoder side)",
            "T_src = 6                                            # source sequence length (encoder side)",
            "d_model = 4                                          # shared embedding width",
            "X_tgt : [seq, d] = randn(0, [T_tgt, d_model])        # target tokens",
            "X_src : [seq, d] = randn(1, [T_src, d_model])        # source tokens (in a real pipeline, encoder output)",
            "Wq = randn(2, [d_model, d_model])                    # Q projection (target side)",
            "Wk = randn(3, [d_model, d_model])                    # K projection (source side)",
            "Wv = randn(4, [d_model, d_model])                    # V projection (source side)",
            "Q = matmul(X_tgt, Wq)                                # queries from target",
            "K = matmul(X_src, Wk)                                # keys from source",
            "V = matmul(X_src, Wv)                                # values from source",
            "scores = matmul(Q, transpose(K)) / sqrt(d_model)     # [T_tgt, T_src] -- non-square: cross-attn signature",
            "weights = softmax(scores, 1)                         # each target row distributes over source positions",
            "shape(weights)                                       # [T_tgt, T_src] -- not [T, T]",
            "svg(weights, \"heatmap\")                              # the non-square pattern is the giveaway",
            "out = matmul(weights, V)                             # mix source values per target query",
            "shape(out)                                           # back to target shape: [T_tgt, d_model]",
        ],
    },
    Demo {
        name: "Encoder Block",
        intro: "One transformer encoder block built from the model DSL: pre-norm self-attention with a residual, then pre-norm feedforward (linear -> relu -> linear) with another residual. Same shape in, same shape out. The residual connections let gradients flow past the attention and FFN layers; the pre-norm `rms_norm` keeps activations stable.",
        takeaway: "An encoder block is just chain(residual(self-attn), residual(ffn)). Stack a dozen of these and you have BERT; add cross-attention and a causal mask and you have a decoder block. The attention heatmap shows what each position is attending to in this single layer; deep encoders compose these patterns into hierarchical representations.",
        lines: &[
            "X = randn(0, [4, 8])                                                                                                                                                # 4 tokens of dim 8",
            "encoder = chain(residual(chain(rms_norm(8), attention(8, 1, 1))), residual(chain(rms_norm(8), linear(8, 16, 2), relu_layer(), linear(16, 8, 3))))  # pre-norm self-attn + residual; pre-norm FFN + residual",
            "out = apply(encoder, X)                                                                                                                                            # forward pass through both sub-blocks",
            "shape(out)                                                                                                                                                          # same shape in, same shape out",
            "bare_attn = attention(8, 1, 1)                                                                                                                                      # standalone layer for the attention_weights extractor",
            "svg(attention_weights(bare_attn, X), \"heatmap\")                                                                                                                    # render the [T, T] attention pattern",
        ],
    },
    Demo {
        name: "Decoder Block",
        intro: "One transformer decoder block: three sub-blocks instead of two. Causal self-attention (target attends only to past target tokens), then cross-attention (target queries attend to encoder output -- built from scratch since MLPL has no cross-attention layer primitive), then a feedforward sub-block. X_tgt is the target sequence (T=4); X_src stands in for the encoder's output (T=6) -- a real pipeline would feed the encoder demo's `out` here.",
        takeaway: "A decoder block is encoder-block + cross-attention. The cross-attention heatmap is non-square ([T_tgt, T_src]): target queries distributing attention over encoder positions. Stack a dozen of these and you have GPT (decoder-only, drop the cross-attn step) or T5 (encoder-decoder, keep all three).",
        lines: &[
            "T_tgt = 4                                                                                # target sequence length",
            "T_src = 6                                                                                # source sequence length (encoder output stand-in)",
            "d_model = 8                                                                              # shared embedding width",
            "d_ff = 16                                                                                # FFN hidden width",
            "X_tgt = randn(0, [T_tgt, d_model])                                                       # target tokens",
            "X_src = randn(1, [T_src, d_model])                                                       # source tokens",
            "self_attn = residual(chain(rms_norm(d_model), causal_attention(d_model, 1, 2)))         # sub-block 1: causal self-attn + residual",
            "H = apply(self_attn, X_tgt)                                                              # target after self-attention",
            "pre_xattn = rms_norm(d_model)                                                            # cross-attn pre-norm layer",
            "H_norm = apply(pre_xattn, H)                                                             # normalized H for cross-attn queries",
            "Wq = randn(3, [d_model, d_model])                                                        # cross-attn Q (target side)",
            "Wk = randn(4, [d_model, d_model])                                                        # cross-attn K (source side)",
            "Wv = randn(5, [d_model, d_model])                                                        # cross-attn V (source side)",
            "Q = matmul(H_norm, Wq)                                                                   # queries from target",
            "K = matmul(X_src, Wk)                                                                    # keys from source",
            "V = matmul(X_src, Wv)                                                                    # values from source",
            "weights = softmax(matmul(Q, transpose(K)) / sqrt(d_model), 1)                            # cross-attention weights [T_tgt, T_src]",
            "shape(weights)                                                                           # confirm non-square",
            "svg(weights, \"heatmap\")                                                                  # the cross-attn pattern",
            "H2 = H + matmul(weights, V)                                                              # residual: H + cross-attn output",
            "ffn = residual(chain(rms_norm(d_model), linear(d_model, d_ff, 6), relu_layer(), linear(d_ff, d_model, 7)))  # sub-block 3: pre-norm FFN + residual",
            "out = apply(ffn, H2)                                                                     # final decoder block output",
            "shape(out)                                                                               # same shape as X_tgt",
        ],
    },
    Demo {
        name: "Basics",
        intro: "The smallest possible MLPL tour: scalar arithmetic, elementwise array arithmetic with broadcasting, variable binding, and unary negation. If this makes sense, you can read the rest of the demos.",
        takeaway: "Operators apply elementwise; scalars broadcast; variables persist across REPL lines. That's the substrate every other demo builds on.",
        lines: &[
            "1 + 2                       # scalar addition",
            "3 * 4                       # scalar multiplication",
            "10 / 3                      # all numbers are f64; integer division does not exist",
            "[1, 2, 3] + [4, 5, 6]       # elementwise vector add",
            "[1, 2, 3] * 10              # scalar broadcasts across the vector",
            "x = [10, 20, 30]            # bind a vector to x",
            "y = x + 1                   # x is unchanged; y is a new vector",
            "y                           # echo y to see [11, 21, 31]",
            "-[1, 2, 3]                  # unary negation",
        ],
    },
    Demo {
        name: "Decision Boundary",
        intro: "Train a two-weight logistic regression on the AND function (only [1,1] is class 1), then visualize the learned probability surface over a 20x20 grid. The sigmoid's S-curve lives in 2D here, tilted by the weight vector.",
        takeaway: "300 gradient steps on four points produced a linear decision boundary that cleanly separates AND's one positive from its three negatives. The surface SVG shows where the model transitions from 'no' to 'yes' in input space.",
        lines: &[
            "X = [[0,0],[0,1],[1,0],[1,1]]                                                                                                                                                                                                                                                                # AND truth-table inputs",
            "y = [0, 0, 0, 1]                                                                                                                                                                                                                                                                              # only [1,1] is positive",
            "w = zeros([2])                                                                                                                                                                                                                                                                                # 2-element weight vector",
            "b = 0                                                                                                                                                                                                                                                                                          # scalar bias",
            "lr = 1.0                                                                                                                                                                                                                                                                                       # learning rate",
            "n = 4                                                                                                                                                                                                                                                                                          # batch size for averaging",
            "repeat 400 { z = matmul(X, reshape(w, [2, 1])) + b; pred = sigmoid(z); dz = pred - reshape(y, [4, 1]); dw = reshape(matmul(transpose(X), dz), [2]) / n; db = mean(dz); w = w - lr * dw; b = b - lr * db }  # 400 hand-rolled gradient-descent steps",
            "gx = grid([0, 1, 0, 1], 20)                                                                                                                                                                                                                                                                  # 20x20 grid of probe points in input space",
            "gz = matmul(gx, reshape(w, [2, 1])) + b                                                                                                                                                                                                                                                       # forward pass on the grid",
            "gp = sigmoid(reshape(gz, [400]))                                                                                                                                                                                                                                                              # squash to probabilities",
            "surface = reshape(gp, [20, 20])                                                                                                                                                                                                                                                               # back to a 2D surface",
            "tp = [[0,0,0],[0,1,0],[1,0,0],[1,1,1]]                                                                                                                                                                                                                                                        # training points + labels for overlay",
            "svg(surface, \"decision_boundary\", tp)                                                                                                                                                                                                                                                        # render the surface with the four points",
        ],
    },
    Demo {
        name: "K-Means",
        intro: "K-means clustering without loops over points: all distances computed in one matmul, cluster assignments via argmax, and centroid updates via a one-hot-matrix-times-data trick. Three blobs in 2D, three centroids, ten iterations.",
        takeaway: "The final scatter shows points colored by their assigned cluster and the three centroids as a separate plot. Unsupervised -- no labels were passed in; the algorithm discovered the three groups from geometry alone.",
        lines: &[
            "D = blobs(7, 30, [[0, 0], [4, 4], [-4, 4]])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # 90 points across three blobs (with labels)",
            "X = matmul(D, [[1,0],[0,1],[0,0]])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # drop the label column for unsupervised work",
            "C = [[1, 1], [3, 3], [-3, 3]]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       # 3 initial centroids",
            "repeat 10 { sqX = reshape(reduce_add(X*X, 1), [90, 1]); sqC = reshape(reduce_add(C*C, 1), [1, 3]); XC = matmul(X, transpose(C)); dists = matmul(sqX, ones([1, 3])) + matmul(ones([90, 1]), sqC) - 2*XC; clus = argmax(-1 * dists, 1); jj = reshape(iota(3), [3, 1]); ll = reshape(clus, [1, 90]); diff = matmul(jj, ones([1, 90])) - matmul(ones([3, 1]), ll); A = eq(diff, 0); counts = reshape(reduce_add(A, 1), [3, 1]); sums = matmul(A, X); C = sums / matmul(counts, ones([1, 2])) }                                                                                                                                                                                              # 10 iterations of assign + update -- no per-point loop",
            "sqX = reshape(reduce_add(X*X, 1), [90, 1])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # per-point squared norms (final assignment)",
            "sqC = reshape(reduce_add(C*C, 1), [1, 3])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # per-centroid squared norms",
            "XC = matmul(X, transpose(C))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       # cross-products",
            "dists = matmul(sqX, ones([1, 3])) + matmul(ones([90, 1]), sqC) - 2*XC                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # squared-distance matrix [90, 3]",
            "clus = argmax(-1 * dists, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       # nearest centroid per point (argmax of negative distance)",
            "scatter_labeled(X, clus)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # points colored by cluster",
            "svg(C, \"scatter\")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # final centroid positions",
        ],
    },
    Demo {
        name: "Logistic Regression",
        intro: "The hello-world of supervised learning: 2D AND with hand-rolled gradient descent, final-accuracy check, a bar chart of per-point predictions, and a 1D loss sweep along the weight direction showing the minimum the optimizer found.",
        takeaway: "Accuracy prints as 1.0; the bar chart shows confident [0, 0, 0, 1] predictions; the loss curve has a clean bowl shape. This is the sanity-check pattern every classifier should pass.",
        lines: &[
            "X = [[0,0],[0,1],[1,0],[1,1]]                                                                                                                                                                                                                                                                # AND truth-table inputs",
            "y = [0, 0, 0, 1]                                                                                                                                                                                                                                                                              # only [1,1] is positive",
            "w = zeros([2])                                                                                                                                                                                                                                                                                # 2-weight model",
            "b = 0                                                                                                                                                                                                                                                                                          # scalar bias",
            "lr = 1.0                                                                                                                                                                                                                                                                                       # learning rate",
            "n = 4                                                                                                                                                                                                                                                                                          # batch size",
            "repeat 300 { z = matmul(X, reshape(w, [2, 1])) + b; pred = sigmoid(z); dz = pred - reshape(y, [4, 1]); dw = reshape(matmul(transpose(X), dz), [2]) / n; db = mean(dz); w = w - lr * dw; b = b - lr * db }  # 300 hand-rolled gradient-descent steps",
            "pred = sigmoid(matmul(X, reshape(w, [2, 1])) + b)                                                                                                                                                                                                                                              # final predictions",
            "rounded = gt(pred, 0.5)                                                                                                                                                                                                                                                                       # threshold to 0/1",
            "accuracy = mean(eq(reshape(rounded, [4]), y))                                                                                                                                                                                                                                                  # fraction correct",
            "accuracy                                                                                                                                                                                                                                                                                       # prints 1.0 when all four right",
            "svg(reshape(pred, [4]), \"bar\")                                                                                                                                                                                                                                                                # bar chart of per-point predictions",
            "ss = iota(16) / 10                                                                                                                                                                                                                                                                             # 16 scaling factors for the loss sweep",
            "WS2 = matmul(reshape(ss, [16, 1]), reshape(w, [1, 2]))                                                                                                                                                                                                                                         # 16 candidate weight vectors along w's direction",
            "zs = matmul(WS2, transpose(X)) + matmul(reshape(ss * b, [16, 1]), ones([1, 4]))                                                                                                                                                                                                                # logits at each scaling",
            "ps = sigmoid(zs)                                                                                                                                                                                                                                                                                # probabilities at each scaling",
            "ys4 = matmul(ones([16, 1]), reshape(y, [1, 4]))                                                                                                                                                                                                                                                 # broadcast labels across the 16 candidates",
            "losses = reduce_add((ps - ys4) * (ps - ys4), 1) / 4                                                                                                                                                                                                                                            # MSE at each scaling",
            "svg(losses, \"line\")                                                                                                                                                                                                                                                                          # 1D loss curve along the weight direction",
        ],
    },
    Demo {
        name: "Loss Curve",
        intro: "Sweep a single weight across 25 values, compute the MSE loss against a linear target at each one, and plot the result. No training -- just the shape of the loss landscape.",
        takeaway: "A smooth parabolic curve with a clear minimum near the true weight. This is what gradient descent is walking down when you train; seeing the bowl makes the 'minimize the loss' story tangible.",
        lines: &[
            "x = [0, 1, 2, 3, 4]                                # input x-values",
            "y = [0, 2, 4, 6, 8]                                # target y-values (slope 2)",
            "ws = iota(25) / 4 - 1                              # 25 candidate weights from -1 to 5",
            "WS = reshape(ws, [25, 1])                          # column vector of weights",
            "preds = matmul(WS, reshape(x, [1, 5]))             # prediction for each weight x each x-value",
            "YS = matmul(ones([25, 1]), reshape(y, [1, 5]))     # broadcast targets across the 25 weights",
            "errs = preds - YS                                  # residuals",
            "losses = reduce_add(errs * errs, 1) / 5            # MSE per weight",
            "svg(losses, \"line\")                               # the loss bowl",
        ],
    },
    Demo {
        name: "Math Functions",
        intro: "The scalar and elementwise math primitives MLPL inherits from NumPy/APL conventions: exp, log, sqrt, abs, pow, sigmoid, tanh. Works on scalars and arrays uniformly.",
        takeaway: "Every primitive broadcasts over array inputs without a loop. If you know these names from NumPy you already know MLPL's math surface.",
        lines: &[
            "exp(0)                            # exp on a scalar -> 1.0",
            "exp([0, 1, 2])                    # exp broadcasts elementwise",
            "log(exp(1))                       # log inverse of exp",
            "sqrt([4, 9, 16, 25])              # elementwise sqrt",
            "abs([-3, 0, 5])                   # elementwise absolute value",
            "pow([2, 3, 4], 2)                 # elementwise squaring",
            "sigmoid(0)                        # logistic at zero is 0.5",
            "sigmoid([-2, -1, 0, 1, 2])        # full S-curve sample",
            "tanh_fn([-1, 0, 1])               # tanh on a small vector",
        ],
    },
    Demo {
        name: "Matrix Ops",
        intro: "Build a 3x4 matrix from iota, transpose it, read its shape and rank, and sum along both axes. The axis argument to reduce_add is how you go from a 2D tensor to a row-sum or column-sum vector.",
        takeaway: "reshape moves between flat and multi-dimensional views without copying; transpose swaps axes; reduce_add with an axis drops that axis. This is the APL half of MLPL -- shape is first-class and cheap to manipulate.",
        lines: &[
            "x = iota(12)                       # flat 0..11 vector",
            "m = reshape(x, [3, 4])             # reshape to a 3x4 matrix",
            "m                                  # display the matrix",
            "transpose(m)                       # swap to 4x3",
            "shape(m)                           # the dimension vector [3, 4]",
            "rank(m)                            # number of dimensions (2)",
            "reduce_add(m, 0)                   # column sums (length 4)",
            "reduce_add(m, 1)                   # row sums (length 3)",
        ],
    },
    Demo {
        name: "Moons MLP",
        intro: "A two-layer tanh MLP trained with Adam + cross-entropy on the 'two moons' synthetic dataset -- two interleaved half-circles that a single linear classifier can't separate. 200 training steps, then accuracy, a loss curve, a confusion matrix, and the learned decision boundary over a 30x30 grid.",
        takeaway: "A non-linear classifier with ~50 parameters separates the moons. cross_entropy(logits, y) is the canonical classification loss and is differentiable end-to-end through grad. The boundary SVG curves around each moon -- a visual proof that a hidden layer plus a non-linearity learned a non-linear decision surface.",
        lines: &[
            "M = moons(7, 120, 0.08)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # 120 points in two interleaved half-circles",
            "X = matmul(M, [[1,0],[0,1],[0,0]])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # drop the label column",
            "y = reshape(matmul(M, [[0],[0],[1]]), [120])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # extract integer labels",
            "O120 = ones([120, 1])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # ones-column for bias broadcasting",
            "W1 = param[2, 8]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # input -> hidden weight matrix (declares trainable)",
            "b1 = param[1, 8]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # hidden bias",
            "W2 = param[8, 2]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # hidden -> output weights",
            "b2 = param[1, 2]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # output bias",
            "W1 = randn(11, [2, 8]) * 0.5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # initialize W1 with seeded normal",
            "W2 = randn(12, [8, 2]) * 0.5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # initialize W2 with seeded normal",
            "train 200 { adam(cross_entropy(matmul(tanh_fn(matmul(X, W1) + matmul(O120, b1)), W2) + matmul(O120, b2), y), [W1, b1, W2, b2], 0.05, 0.9, 0.999, 0.00000001); cross_entropy(matmul(tanh_fn(matmul(X, W1) + matmul(O120, b1)), W2) + matmul(O120, b2), y) }  # 200 Adam steps with cross-entropy loss",
            "Z1 = matmul(X, W1) + matmul(O120, b1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # forward: hidden pre-activation",
            "H = tanh_fn(Z1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # hidden activation",
            "Z2 = matmul(H, W2) + matmul(O120, b2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # output pre-activation",
            "P = softmax(Z2, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # row-wise probabilities",
            "pred = argmax(P, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                # discrete class predictions",
            "accuracy = mean(eq(pred, y))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       # fraction correct",
            "accuracy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # prints close to 1.0",
            "loss_curve(last_losses)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # train block auto-captured per-step losses",
            "confusion_matrix(pred, y)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # 2x2 prediction-vs-actual",
            "G = grid([-1.5, 2.5, -1, 1.5], 30)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # 30x30 grid in input space",
            "O900 = ones([900, 1])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # broadcast helper for grid bias",
            "GZ1 = matmul(G, W1) + matmul(O900, b1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # grid forward: hidden pre-activation",
            "GH = tanh_fn(GZ1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # grid hidden activation",
            "GZ2 = matmul(GH, W2) + matmul(O900, b2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # grid output pre-activation",
            "GP = softmax(GZ2, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                # grid probabilities",
            "p1 = reshape(matmul(GP, [[0],[1]]), [900])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # extract probability of class 1",
            "boundary_2d(p1, [30, 30], X, y)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      # render the curved decision surface",
        ],
    },
    Demo {
        name: "PCA",
        intro: "Principal Component Analysis without calling into a library: make anisotropic 2D data, center it, form the covariance matrix, run power iteration to find the top eigenvector, and project every point onto it. The top axis is the direction of maximum variance.",
        takeaway: "The scatter is colored by which side of zero each point's projection lands on; the line shows the found principal axis. Power iteration converges in ~10 steps to a direction that's clearly the long axis of the data cloud.",
        lines: &[
            "Xraw = randn(1, [60, 2])                                          # 60 points of isotropic 2D noise",
            "X = matmul(Xraw, [[1, 2], [0, 0.3]])                              # stretch into an anisotropic cloud",
            "cm = reduce_add(X, 0) / 60                                        # column means -- the centroid",
            "Xc = X - matmul(ones([60, 1]), reshape(cm, [1, 2]))               # center the data",
            "Cov = matmul(transpose(Xc), Xc) / 60                              # 2x2 covariance matrix",
            "v = [1, 0]                                                         # initial guess for the top eigenvector",
            "repeat 10 { v = matmul(Cov, v); v = v / sqrt(dot(v, v)) }         # power iteration: 10 multiplies + normalize",
            "coords = reshape(matmul(Xc, reshape(v, [2, 1])), [60])            # project each point onto the principal axis",
            "labels = gt(coords, 0)                                             # which side of the axis each point lands on",
            "ends = matmul(reshape([-3, 3], [2, 1]), reshape(v, [1, 2]))       # endpoints of the axis line in centered space",
            "line = ends + matmul(ones([2, 1]), reshape(cm, [1, 2]))           # shift the line back into original space",
            "scatter_labeled(X, labels)                                         # data colored by side of the principal axis",
            "svg(line, \"line\")                                                 # the principal axis itself",
        ],
    },
    Demo {
        name: "Softmax Classifier",
        intro: "Three-class classification on 90 blob points. softmax + cross-entropy gradient descent (done in closed form here -- P - Y IS the gradient through the softmax+CE pair), then the usual accuracy / confusion-matrix / decision-boundary close.",
        takeaway: "A single linear layer with three output heads separates three well-spaced clusters. The decision boundary shows three roughly-wedge-shaped regions, one per class.",
        lines: &[
            "D = blobs(11, 30, [[0, 0], [4, 4], [-4, 4]])                                                                                                                                                                                                                                                  # 90 points in three blobs",
            "X = matmul(D, [[1,0],[0,1],[0,0]])                                                                                                                                                                                                                                                            # drop the label column",
            "tl = reshape(matmul(D, [[0],[0],[1]]), [90])                                                                                                                                                                                                                                                  # integer labels (length 90)",
            "Y = one_hot(tl, 3)                                                                                                                                                                                                                                                                            # one-hot targets [90, 3]",
            "W = zeros([2, 3])                                                                                                                                                                                                                                                                              # 2-input, 3-class weight matrix",
            "b = zeros([3])                                                                                                                                                                                                                                                                                  # per-class bias",
            "lr = 0.2                                                                                                                                                                                                                                                                                       # learning rate",
            "repeat 300 { logits = matmul(X, W) + matmul(ones([90, 1]), reshape(b, [1, 3])); P = softmax(logits, 1); dZ = P - Y; gW = matmul(transpose(X), dZ) / 90; gb = reduce_add(dZ, 0) / 90; W = W - lr * gW; b = b - lr * gb }  # 300 closed-form gradient steps (P-Y is the softmax+CE gradient)",
            "logits = matmul(X, W) + matmul(ones([90, 1]), reshape(b, [1, 3]))                                                                                                                                                                                                                              # final logits",
            "P = softmax(logits, 1)                                                                                                                                                                                                                                                                          # final probabilities",
            "pred = argmax(P, 1)                                                                                                                                                                                                                                                                             # class predictions",
            "accuracy = mean(eq(pred, tl))                                                                                                                                                                                                                                                                   # fraction correct",
            "accuracy                                                                                                                                                                                                                                                                                        # prints close to 1.0",
            "confusion_matrix(pred, tl)                                                                                                                                                                                                                                                                      # 3x3 prediction-vs-actual",
            "G = grid([-6, 6, -3, 6], 30)                                                                                                                                                                                                                                                                   # 30x30 grid in input space",
            "gl = matmul(G, W) + matmul(ones([900, 1]), reshape(b, [1, 3]))                                                                                                                                                                                                                                 # logits over the grid",
            "gP = softmax(gl, 1)                                                                                                                                                                                                                                                                             # per-grid-cell class probabilities",
            "p0 = reshape(matmul(gP, [[1],[0],[0]]), [900])                                                                                                                                                                                                                                                  # extract probability of class 0",
            "boundary_2d(p0, [30, 30], X, tl)                                                                                                                                                                                                                                                                # three-wedge decision surface",
        ],
    },
    // The interactive `Tiny LM` and `Tiny LM Generate` demos use a
    // smaller configuration than `demos/tiny_lm.mlpl` (V=280, d=32,
    // 200 steps) -- the CLI source is tuned for a serious run under
    // a native build where each training step is milliseconds. The
    // interpreter + WASM stack on the browser main thread is ~100x
    // slower per op; running the CLI configuration here hangs the
    // event loop past the browser's "unresponsive tab" timeout.
    // These entries mirror the "Training and Generating" tutorial
    // lesson's budget (V=260, d=16, block=8, 30 steps) which
    // completes in a few seconds interactively. For the serious
    // run, use the CLI: `mlpl-repl -f demos/tiny_lm.mlpl`.
    Demo {
        name: "Tiny LM Generate",
        intro: "End-to-end tiny transformer language model: tokenize a small corpus with BPE, wrap inputs as next-token pairs, build a 1-layer model (embed + causal_attention + rms_norm + linear head), train 30 Adam steps, then sample 20 tokens from a prompt and render the attention heatmap. The smallest program that trains and generates text.",
        takeaway: "A language model with a few thousand parameters produced a decoded string from a learned distribution and a [T,T] heatmap of what each position attended to. The loss curve should trend downward; the generated text is noisy because the model and budget are both tiny -- scale V, d, block, and step count for a serious run via the CLI.",
        lines: &[
            "corpus = load_preloaded(\"tiny_corpus\")                                                                                                                                                                                                                                                          # short pangram-style training corpus",
            "tok    = train_bpe(corpus, 260, 0)                                                                                                                                                                                                                                                                # train a 260-token BPE tokenizer",
            "ids    = apply_tokenizer(tok, corpus)                                                                                                                                                                                                                                                              # encode corpus -> integer ids",
            "X_all = shift_pairs_x(ids, 8)                                                                                                                                                                                                                                                                       # next-token input windows of length 8",
            "Y_all = shift_pairs_y(ids, 8)                                                                                                                                                                                                                                                                       # matching target windows shifted by 1",
            "X     = reshape(X_all, [reduce_mul(shape(X_all))])                                                                                                                                                                                                                                                  # flatten to 1-D for the embed lookup",
            "Y     = reshape(Y_all, [reduce_mul(shape(Y_all))])                                                                                                                                                                                                                                                  # flatten targets",
            "V = 260 ; d = 16 ; h = 1                                                                                                                                                                                                                                                                            # vocab / embed-dim / heads",
            "model = chain(embed(V, d, 0), causal_attention(d, h, 1), rms_norm(d), linear(d, V, 2))                                                                                                                                                                                                              # 1-layer transformer LM",
            "experiment \"tiny_lm_gen\" { train 30 { adam(cross_entropy(apply(model, X), Y), model, 0.01, 0.9, 0.999, 0.00000001); loss_metric = cross_entropy(apply(model, X), Y) } }  # train 30 Adam steps; record per-step loss",
            "loss_curve(last_losses)                                                                                                                                                                                                                                                                             # plot the training loss",
            "prompt = apply_tokenizer(tok, \"the \")                                                                                                                                                                                                                                                              # seed prompt as token ids",
            "seq    = prompt                                                                                                                                                                                                                                                                                      # generation buffer",
            "repeat 20 { logits = apply(model, seq); last = last_row(logits); nxt = sample(top_k(last, 20), 0.8, step); seq = concat(seq, nxt) }                                                                                                                                                                  # 20-token sample with top-k 20 + temperature 0.8",
            "decode(tok, seq)                                                                                                                                                                                                                                                                                     # decode ids back to text",
            "viz_ids = apply_tokenizer(tok, \"the quick\")                                                                                                                                                                                                                                                          # short input for the heatmap",
            "attn_w  = attention_weights(model, viz_ids)                                                                                                                                                                                                                                                          # extract the [T, T] attention weights",
            "svg(attn_w, \"heatmap\")                                                                                                                                                                                                                                                                              # what each position attended to",
        ],
    },
    Demo {
        name: "Tiny LM",
        intro: "Training-only variant of the Tiny LM: same model shape (embed + causal_attention + rms_norm + linear head, V=260, d=16, block=8), wrapped in experiment so the final metric is logged, then plot the loss curve across 30 Adam steps. No generation -- just a clean 'does the loss go down' demo.",
        takeaway: "Loss curve trends downward over 30 steps; the :experiments registry records the run's final loss metric. For generation + attention visualization, see 'Tiny LM Generate'. For a serious run with V=280 and 200 steps, use the CLI: mlpl-repl -f demos/tiny_lm.mlpl.",
        lines: &[
            "corpus = load_preloaded(\"tiny_corpus\")                                                                                                                                                                                                                                                          # built-in pangram corpus",
            "tok    = train_bpe(corpus, 260, 0)                                                                                                                                                                                                                                                                # train a 260-token BPE tokenizer",
            "ids    = apply_tokenizer(tok, corpus)                                                                                                                                                                                                                                                              # encode the corpus to ids",
            "X_all = shift_pairs_x(ids, 8)                                                                                                                                                                                                                                                                       # next-token input windows of length 8",
            "Y_all = shift_pairs_y(ids, 8)                                                                                                                                                                                                                                                                       # matching target windows shifted by 1",
            "X     = reshape(X_all, [reduce_mul(shape(X_all))])                                                                                                                                                                                                                                                  # flatten to 1-D",
            "Y     = reshape(Y_all, [reduce_mul(shape(Y_all))])                                                                                                                                                                                                                                                  # flatten targets",
            "V = 260 ; d = 16 ; h = 1                                                                                                                                                                                                                                                                            # vocab / embed-dim / heads",
            "model = chain(embed(V, d, 0), causal_attention(d, h, 1), rms_norm(d), linear(d, V, 2))                                                                                                                                                                                                              # 1-layer transformer LM",
            "experiment \"tiny_lm\" { train 30 { adam(cross_entropy(apply(model, X), Y), model, 0.01, 0.9, 0.999, 0.00000001); loss_metric = cross_entropy(apply(model, X), Y) } }  # 30 Adam steps inside an experiment block",
            "loss_curve(last_losses)                                                                                                                                                                                                                                                                             # plot the training loss",
        ],
    },
    Demo {
        name: "Tiny MLP",
        intro: "A hand-rolled 2-layer MLP on four blobs (two per class, XOR-ish layout). Forward + backward pass written out explicitly so every gradient is visible -- no autograd, no Model DSL. Ends with the decision-boundary render over a 30x30 grid.",
        takeaway: "You can see every matmul and every elementwise op in the training loop. For the same model with autograd hiding the backward pass, see 'Moons MLP'; for the same model as a Model DSL composition, see the tutorial's 'Model Composition' lesson.",
        lines: &[
            "D = blobs(3, 20, [[-2,-2],[2,2],[-2,2],[2,-2]])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # 80 points in four blobs (XOR layout)",
            "X = matmul(D, [[1,0],[0,1],[0,0]])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # drop the label column",
            "raw = reshape(matmul(D, [[0],[0],[1]]), [80])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # raw labels 0..3 (one per quadrant)",
            "y = gt(raw, 1.5)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # binary label: half the quadrants are positive",
            "Y = one_hot(y, 2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # one-hot for the loss",
            "W1 = randn(5, [2, 8]) * 0.5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # input -> hidden weights",
            "b1 = zeros([8])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # hidden bias",
            "W2 = randn(6, [8, 2]) * 0.5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # hidden -> output weights",
            "b2 = zeros([2])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # output bias",
            "lr = 0.2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # learning rate",
            "repeat 600 { Z1 = matmul(X, W1) + matmul(ones([80, 1]), reshape(b1, [1, 8])); H = tanh_fn(Z1); Z2 = matmul(H, W2) + matmul(ones([80, 1]), reshape(b2, [1, 2])); P = softmax(Z2, 1); dZ2 = P - Y; gW2 = matmul(transpose(H), dZ2) / 80; gb2 = reduce_add(dZ2, 0) / 80; dH = matmul(dZ2, transpose(W2)); dZ1 = dH * (1 - H * H); gW1 = matmul(transpose(X), dZ1) / 80; gb1 = reduce_add(dZ1, 0) / 80; W1 = W1 - lr * gW1; b1 = b1 - lr * gb1; W2 = W2 - lr * gW2; b2 = b2 - lr * gb2 }  # 600 hand-rolled forward + backward steps; chain rule visible in dZ1 = dH * (1 - H*H)",
            "Z1 = matmul(X, W1) + matmul(ones([80, 1]), reshape(b1, [1, 8]))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # final forward: hidden pre-activation",
            "H = tanh_fn(Z1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # hidden activation",
            "Z2 = matmul(H, W2) + matmul(ones([80, 1]), reshape(b2, [1, 2]))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # output pre-activation",
            "P = softmax(Z2, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # row-wise probabilities",
            "mlp_pred = argmax(P, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # discrete class predictions",
            "mean(eq(mlp_pred, y))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                # accuracy",
            "confusion_matrix(mlp_pred, y)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # 2x2 prediction-vs-actual",
            "G = grid([-4, 4, -4, 4], 30)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # 30x30 input-space grid",
            "GZ1 = matmul(G, W1) + matmul(ones([900, 1]), reshape(b1, [1, 8]))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # grid forward: hidden pre-activation",
            "GH = tanh_fn(GZ1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # grid hidden activation",
            "GZ2 = matmul(GH, W2) + matmul(ones([900, 1]), reshape(b2, [1, 2]))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # grid output pre-activation",
            "GP = softmax(GZ2, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # grid probabilities",
            "p1 = reshape(matmul(GP, [[0],[1]]), [900])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # extract probability of class 1",
            "boundary_2d(p1, [30, 30], X, y)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # the decision surface that XOR needs",
        ],
    },
    Demo {
        name: "Workspace Introspection",
        intro: "Tour of the REPL's introspection commands: :version, :wsid (workspace ID summary), :vars, :describe, :models, :experiments, :fns. Also shows how the axis-label annotation syntax (M : [batch, feat] = ...) shows up in :vars and :describe output.",
        takeaway: "You can always ask the REPL what's in your session. :describe on a variable prints shape + labels + a preview; on a model, the layer tree; on a builtin, the signature and one-line doc. :experiments shows every tracked run.",
        lines: &[
            ":version                                                            # build banner: version + arch + commit + timestamp",
            ":wsid                                                                # workspace summary (var/param/model counts)",
            "x = 42                                                               # bind a scalar",
            "v = iota(5)                                                          # bind a 5-vector",
            "M : [batch, feat] = reshape(iota(6), [2, 3])                         # bind a labeled-axis matrix",
            ":vars                                                                # list all bound variables with shape + tag",
            ":describe v                                                          # shape + values preview for a variable",
            "mdl = chain(linear(2, 4, 11), relu_layer(), linear(4, 2, 12))        # bind a model",
            ":models                                                              # list bound models with layer trees",
            ":describe mdl                                                        # the layer tree for mdl",
            "tok = train_bpe(\"abababab\", 260, 0)                                  # bind a tokenizer",
            ":describe tok                                                        # tokenizer info",
            "W = param[3, 2]                                                      # declare a trainable parameter",
            ":vars                                                                # see the [param] tag on W",
            ":wsid                                                                # parameter count went up",
            "experiment \"workspace_demo\" { loss_metric = 0.25; accuracy_metric = 0.94 }  # capture two metrics into the experiment registry",
            ":experiments                                                         # list every captured run",
            ":describe matmul                                                     # signature + one-line doc for a builtin",
            ":fns                                                                 # list user-defined functions (none yet)",
        ],
    },
    Demo {
        name: "Visualizations",
        intro: "The four primitive svg() types -- scatter, line, bar, heatmap -- each in one line. Rendered inline; a download button next to each SVG saves it as a file.",
        takeaway: "Every plot is one call with a data array and a type string. There is no plotting API to learn beyond 'pass the right shape.'",
        lines: &[
            "svg([[0,0],[1,1],[2,4],[3,9],[4,16]], \"scatter\")    # Nx2 -> circle per row",
            "svg([1, 3, 2, 5, 4, 6], \"line\")                      # vector -> polyline",
            "svg([3, 1, 4, 1, 5, 9, 2, 6], \"bar\")                 # vector -> bar chart",
            "svg(reshape(iota(25), [5, 5]), \"heatmap\")            # MxN -> viridis grid",
        ],
    },
    Demo {
        name: "ViT Attention Pattern (no training)",
        intro: "Vision Transformer pipeline end-to-end on a synthetic 64x64 RGB image, no training. patchify(x, 16) cuts the image into a 4x4 grid of 16 patches; each patch flattens P*P*C = 768 floats. A linear projection embeds them to d_model=128, a randn CLS token is prepended via concat(cls, patches, 0), a randn positional embedding is added, and one attention head computes its softmax weight matrix [17, 17] (16 patch tokens + 1 CLS). The heatmap is what an UNTRAINED ViT head pays attention to -- essentially nothing useful. Demo 2 (Saga 29 step 007, coming next) trains the same architecture on pets_tiny.",
        takeaway: "The ViT recipe is shorter than it looks: patchify + linear-embed + CLS-prepend + pos-add + attention. Five new builtins (patchify, concat axis 0, randn, matmul, softmax) compose into the same forward-pass topology the original ViT paper drew. Without training, the attention weights are random; the row-sum sanity check at the end confirms softmax did its job regardless.",
        lines: &[
            "img = randn(101, [1, 3, 64, 64])                       # one synthetic 64x64 RGB image",
            "tokens_4d = patchify(img, 16)                          # [1, 16, 768] -- 4x4 grid of P*P*C patches",
            "tokens    = reshape(tokens_4d, [16, 768])              # drop the singleton batch dim",
            "Wp        = randn(201, [768, 128]) / sqrt(768)         # linear patch embedding to d_model = 128",
            "patches   = matmul(tokens, Wp)                         # [16, 128]",
            "cls = randn(301, [1, 128])                             # learnable CLS token (randn for the no-training demo)",
            "seq_no_pos = concat(cls, patches, 0)                   # prepend CLS along the sequence axis -> [17, 128]",
            "pos = randn(401, [17, 128])                            # positional embedding",
            "seq = seq_no_pos + pos                                 # add positions",
            "Wq = randn(501, [128, 128]) / sqrt(128)                # one attention head, d_k = 128",
            "Wk = randn(502, [128, 128]) / sqrt(128)",
            "q  = matmul(seq, Wq)",
            "k  = matmul(seq, Wk)",
            "scores = matmul(q, transpose(k)) / sqrt(128)           # scaled dot-product",
            "A = softmax(scores, 1)                                 # row-stochastic attention pattern [17, 17]",
            "svg(A, \"heatmap\")                                    # render the heatmap",
            "row_sums = reduce_add(A, 1)                            # sanity: every row sums to 1.0",
            "row_sums",
        ],
    },
    Demo {
        name: "Pets: cat vs dog (quick)",
        intro: "Trained Vision Transformer end-to-end on a balanced 8-image subset of pets_tiny (4 cats + 4 dogs). 30 full-batch adam steps with single-head attention(128, 1) and a tiny MLP classifier head. Saga 29 step 009. The longer 20-image / 100-step version lives in demos/vit_single_head_quick.mlpl; this in-browser variant is scaled down so the train loop completes in seconds.",
        takeaway: "ViT in MLPL is the same five-step recipe as the no-training demo plus a train{} block and an adam call. The forward expression has to be inlined into adam because grad walks the expression tree (let-bindings would create non-differentiable leaves). First-token pooling stands in for a learned CLS token. The loss_curve at the end shows binary cross-entropy falling from ~0.69 (random) toward 0 (overfit); the final accuracy on the 8 training images is the scalar at the very bottom.",
        lines: &[
            "# Balanced 8-image subset (4 cats + 4 dogs) extracted",
            "# from pets_tiny via take + concat.",
            "pets = load_preloaded(\"pets_tiny\")",
            "c0 = reshape(take(pets.X, 0, 0), [1, 3, 64, 64])",
            "c1 = reshape(take(pets.X, 0, 1), [1, 3, 64, 64])",
            "c2 = reshape(take(pets.X, 0, 2), [1, 3, 64, 64])",
            "c3 = reshape(take(pets.X, 0, 3), [1, 3, 64, 64])",
            "d0 = reshape(take(pets.X, 0, 100), [1, 3, 64, 64])",
            "d1 = reshape(take(pets.X, 0, 101), [1, 3, 64, 64])",
            "d2 = reshape(take(pets.X, 0, 102), [1, 3, 64, 64])",
            "d3 = reshape(take(pets.X, 0, 103), [1, 3, 64, 64])",
            "cats = concat(concat(concat(c0, c1, 0), c2, 0), c3, 0)",
            "dogs = concat(concat(concat(d0, d1, 0), d2, 0), d3, 0)",
            "X    = concat(cats, dogs, 0)                                # [8, 3, 64, 64]",
            "ly0 = reshape(take(pets.Y, 0, 0),   [1])",
            "ly1 = reshape(take(pets.Y, 0, 1),   [1])",
            "ly2 = reshape(take(pets.Y, 0, 2),   [1])",
            "ly3 = reshape(take(pets.Y, 0, 3),   [1])",
            "ld0 = reshape(take(pets.Y, 0, 100), [1])",
            "ld1 = reshape(take(pets.Y, 0, 101), [1])",
            "ld2 = reshape(take(pets.Y, 0, 102), [1])",
            "ld3 = reshape(take(pets.Y, 0, 103), [1])",
            "ycats = concat(concat(concat(ly0, ly1, 0), ly2, 0), ly3, 0)",
            "ydogs = concat(concat(concat(ld0, ld1, 0), ld2, 0), ld3, 0)",
            "Y     = concat(ycats, ydogs, 0)                             # [8] -- 0=cat, 1=dog",
            "# Model layers.",
            "linear_p   = linear(768, 128, 17)                            # patch embedding",
            "attn       = attention(128, 1, 23)                           # single-head, d_model=128",
            "classifier = chain(linear(128, 64, 31), relu_layer(), linear(64, 2, 37))",
            "# 30 full-batch adam steps. Forward pipeline:",
            "#   patchify(X, 16) -> reshape -> linear -> reshape",
            "#   -> attention(rank-3) -> take(_, 1, 0) (first-token)",
            "#   -> classifier -> cross_entropy(_, Y)",
            "train 30 { adam(cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [128, 768])), [8, 16, 128])), 1, 0)), Y), [linear_p, attn, classifier], 0.001, 0.9, 0.999, 0.00000001); cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [128, 768])), [8, 16, 128])), 1, 0)), Y) }",
            "# Loss curve: cross-entropy per step.",
            "loss_curve(last_losses)",
            "# Final training accuracy on the 8-image set.",
            "logits   = apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [128, 768])), [8, 16, 128])), 1, 0))",
            "preds    = argmax(logits, 1)",
            "correct  = eq(preds, Y)",
            "accuracy = reduce_add(correct) / 8",
            "accuracy",
            "# Inspecting the tensor: pets.X is [200, 3, 64, 64] --",
            "# too big to print in full. The REPL summarizes it as",
            "# <DenseArray shape=...> with a 3-value preview. To",
            "# peek at actual values, chain `take` down to <= 64",
            "# elements:",
            "take(pets.X, 0, 0)                          # one image -> [3, 64, 64] (still summarized)",
            "take(take(pets.X, 0, 0), 0, 0)              # one channel -> [64, 64] (still summarized)",
            "take(take(take(pets.X, 0, 0), 0, 0), 0, 30) # one row of one channel -> [64] (prints in full)",
            "take(take(take(take(pets.X, 0, 0), 0, 0), 0, 30), 0, 30) # single pixel value -> scalar",
            "# preds and Y on the trained 8-image batch:",
            "preds",
            "Y",
        ],
    },
    Demo {
        name: "Pets: predict + gallery",
        intro: "Train a Vision Transformer on 16 balanced cat/dog images, then render the labeled gallery: each thumbnail gets an `actual / predicted` caption underneath. Saga 29 step 011. Heavy: about 1-3 minutes in the browser. Misclassifications stand out because the two captions disagree (0/1 or 1/0).",
        takeaway: "The new pieces vs the quick demo: `predict_batch(model, X)` runs forward and returns argmax integer labels in one call; `svg(images, \"gallery\", overlay)` accepts an optional `[N]` or `[N, K]` overlay tensor and renders the values as text under each thumbnail. With actual + predicted side-by-side, you can finally point at a specific cat/dog photo and see what the model said. The 50-step train loop overfits, so almost every prediction matches the actual; the demonstration is the visualization, not the model quality.",
        lines: &[
            "pets = load_preloaded(\"pets_tiny\")",
            "c0 = reshape(take(pets.X, 0, 0), [1, 3, 64, 64])",
            "c1 = reshape(take(pets.X, 0, 1), [1, 3, 64, 64])",
            "c2 = reshape(take(pets.X, 0, 2), [1, 3, 64, 64])",
            "c3 = reshape(take(pets.X, 0, 3), [1, 3, 64, 64])",
            "c4 = reshape(take(pets.X, 0, 4), [1, 3, 64, 64])",
            "c5 = reshape(take(pets.X, 0, 5), [1, 3, 64, 64])",
            "c6 = reshape(take(pets.X, 0, 6), [1, 3, 64, 64])",
            "c7 = reshape(take(pets.X, 0, 7), [1, 3, 64, 64])",
            "d0 = reshape(take(pets.X, 0, 100), [1, 3, 64, 64])",
            "d1 = reshape(take(pets.X, 0, 101), [1, 3, 64, 64])",
            "d2 = reshape(take(pets.X, 0, 102), [1, 3, 64, 64])",
            "d3 = reshape(take(pets.X, 0, 103), [1, 3, 64, 64])",
            "d4 = reshape(take(pets.X, 0, 104), [1, 3, 64, 64])",
            "d5 = reshape(take(pets.X, 0, 105), [1, 3, 64, 64])",
            "d6 = reshape(take(pets.X, 0, 106), [1, 3, 64, 64])",
            "d7 = reshape(take(pets.X, 0, 107), [1, 3, 64, 64])",
            "cats_2 = concat(c0, c1, 0)",
            "cats_4 = concat(concat(cats_2, c2, 0), c3, 0)",
            "cats_6 = concat(concat(cats_4, c4, 0), c5, 0)",
            "cats   = concat(concat(cats_6, c6, 0), c7, 0)",
            "dogs_2 = concat(d0, d1, 0)",
            "dogs_4 = concat(concat(dogs_2, d2, 0), d3, 0)",
            "dogs_6 = concat(concat(dogs_4, d4, 0), d5, 0)",
            "dogs   = concat(concat(dogs_6, d6, 0), d7, 0)",
            "X = concat(cats, dogs, 0)",
            "ly0 = reshape(take(pets.Y, 0, 0),   [1])",
            "ly1 = reshape(take(pets.Y, 0, 1),   [1])",
            "ly2 = reshape(take(pets.Y, 0, 2),   [1])",
            "ly3 = reshape(take(pets.Y, 0, 3),   [1])",
            "ly4 = reshape(take(pets.Y, 0, 4),   [1])",
            "ly5 = reshape(take(pets.Y, 0, 5),   [1])",
            "ly6 = reshape(take(pets.Y, 0, 6),   [1])",
            "ly7 = reshape(take(pets.Y, 0, 7),   [1])",
            "ld0 = reshape(take(pets.Y, 0, 100), [1])",
            "ld1 = reshape(take(pets.Y, 0, 101), [1])",
            "ld2 = reshape(take(pets.Y, 0, 102), [1])",
            "ld3 = reshape(take(pets.Y, 0, 103), [1])",
            "ld4 = reshape(take(pets.Y, 0, 104), [1])",
            "ld5 = reshape(take(pets.Y, 0, 105), [1])",
            "ld6 = reshape(take(pets.Y, 0, 106), [1])",
            "ld7 = reshape(take(pets.Y, 0, 107), [1])",
            "yca_2 = concat(ly0, ly1, 0)",
            "yca_4 = concat(concat(yca_2, ly2, 0), ly3, 0)",
            "yca_6 = concat(concat(yca_4, ly4, 0), ly5, 0)",
            "yca   = concat(concat(yca_6, ly6, 0), ly7, 0)",
            "ydo_2 = concat(ld0, ld1, 0)",
            "ydo_4 = concat(concat(ydo_2, ld2, 0), ld3, 0)",
            "ydo_6 = concat(concat(ydo_4, ld4, 0), ld5, 0)",
            "ydo   = concat(concat(ydo_6, ld6, 0), ld7, 0)",
            "Y = concat(yca, ydo, 0)",
            "# Model: shared with the quick demo.",
            "linear_p   = linear(768, 128, 17)",
            "attn       = attention(128, 1, 23)",
            "classifier = chain(linear(128, 64, 31), relu_layer(), linear(64, 2, 37))",
            "# Training loop split into 6 chunks of 5 steps. Each",
            "# chunk runs synchronously in WASM (~5s); the demo runner",
            "# yields to the browser between chunks so the tab stays",
            "# responsive and you can see incremental progress. Adam's",
            "# momentum state persists across chunks via env, so this",
            "# is mathematically equivalent to a single train 30.",
            "train 5 { adam(cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0)), Y), [linear_p, attn, classifier], 0.001, 0.9, 0.999, 0.00000001); cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0)), Y) }",
            "all_losses = last_losses",
            "train 5 { adam(cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0)), Y), [linear_p, attn, classifier], 0.001, 0.9, 0.999, 0.00000001); cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0)), Y) }",
            "all_losses = concat(all_losses, last_losses, 0)",
            "train 5 { adam(cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0)), Y), [linear_p, attn, classifier], 0.001, 0.9, 0.999, 0.00000001); cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0)), Y) }",
            "all_losses = concat(all_losses, last_losses, 0)",
            "train 5 { adam(cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0)), Y), [linear_p, attn, classifier], 0.001, 0.9, 0.999, 0.00000001); cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0)), Y) }",
            "all_losses = concat(all_losses, last_losses, 0)",
            "train 5 { adam(cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0)), Y), [linear_p, attn, classifier], 0.001, 0.9, 0.999, 0.00000001); cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0)), Y) }",
            "all_losses = concat(all_losses, last_losses, 0)",
            "train 5 { adam(cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0)), Y), [linear_p, attn, classifier], 0.001, 0.9, 0.999, 0.00000001); cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0)), Y) }",
            "all_losses = concat(all_losses, last_losses, 0)",
            "loss_curve(all_losses)",
            "# Predict labels then render the labeled gallery.",
            "preds = predict_batch(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [256, 768])), [16, 16, 128])), 1, 0))",
            "preds_2col = concat(reshape(Y, [16, 1]), reshape(preds, [16, 1]), 1)",
            "accuracy = reduce_add(eq(preds, Y)) / 16",
            "accuracy",
            "svg(X, \"gallery\", preds_2col)",
        ],
    },
    Demo {
        name: "ViT Multi-Head Attention Pattern (no training)",
        intro: "Saga 29 step 014: the multi-head sibling of the single-head ViT attention pattern demo. Same synthetic 64x64 image, same patchify+linear-embed+CLS+pos pipeline, but `attention(128, 4, ...)` instead of one head. attention_weights returns [4, 17, 17]; the new `heatmap_grid` viz type renders all four heads in a 2x2 grid so per-head differences are visible at a glance. UNTRAINED: all four heads draw from the same randn distribution and look uniform-random with no structure. Compare with the trained companion demo to see what gradient descent buys.",
        takeaway: "Multi-head attention is just single-head attention applied to h independent subspaces of d_model. Before training, the four heads start from independent randn projections and produce four uniformly-random attention patterns. The architecture doesn't tell heads to specialize; the visualization here is the negative control -- the baseline that the trained demo is judged against.",
        lines: &[
            "img = randn(101, [1, 3, 64, 64])                       # one synthetic 64x64 RGB image",
            "tokens_4d = patchify(img, 16)                          # [1, 16, 768]",
            "tokens    = reshape(tokens_4d, [16, 768])",
            "Wp        = randn(201, [768, 128]) / sqrt(768)         # linear patch embedding to d_model = 128",
            "patches   = matmul(tokens, Wp)",
            "cls = randn(301, [1, 128])                             # CLS token",
            "seq_no_pos = concat(cls, patches, 0)                   # [17, 128]",
            "pos = randn(401, [17, 128])",
            "seq = seq_no_pos + pos                                 # [17, 128]",
            "# Four-head attention via the Model DSL. d_k = 128/4 = 32.",
            "mdl = attention(128, 4, 17)",
            "# attention_weights returns [heads, T, T] for multi-head, rank-2 input.",
            "A = attention_weights(mdl, seq)                        # [4, 17, 17]",
            "# heatmap_grid renders [N, R, C] as a 2x2 grid of",
            "# heatmaps with per-cell colormaps. New viz type, Saga 29 step 014.",
            "svg(A, \"heatmap_grid\")",
            "# Sanity: per-(head, row) softmax sums = 1.",
            "row_sums = reduce_add(A, 2)                            # [4, 17]",
            "row_sums",
        ],
    },
    Demo {
        name: "Pets: multi-head ViT (quick + viz)",
        intro: "Saga 29 step 014: trained 4-head Vision Transformer on a balanced 8-image cat/dog subset, with per-head attention maps rendered after training. Same architecture as the single-head quick demo but `attention(128, 4)` and an `attention_weights(...)` + `heatmap_grid` viz at the end. Browser-friendly: 30 adam steps on 8 images. Compare the heatmap_grid output here with the UNTRAINED 'ViT Multi-Head Attention Pattern' demo -- the four heads start identical and end specialized.",
        takeaway: "Multi-head specialization is learned, not architected. The architecture says 'split d_model into 4 subspaces and run attention in each'; gradient descent then chooses what each subspace pays attention to. Run the untrained demo first to see the baseline, then run this one to see the four post-training heatmaps. Different heads end up focusing on different patches; one or two often concentrate on a single patch (sharp distributions) while others remain diffuse.",
        lines: &[
            "pets = load_preloaded(\"pets_tiny\")",
            "c0 = reshape(take(pets.X, 0, 0), [1, 3, 64, 64])",
            "c1 = reshape(take(pets.X, 0, 1), [1, 3, 64, 64])",
            "c2 = reshape(take(pets.X, 0, 2), [1, 3, 64, 64])",
            "c3 = reshape(take(pets.X, 0, 3), [1, 3, 64, 64])",
            "d0 = reshape(take(pets.X, 0, 100), [1, 3, 64, 64])",
            "d1 = reshape(take(pets.X, 0, 101), [1, 3, 64, 64])",
            "d2 = reshape(take(pets.X, 0, 102), [1, 3, 64, 64])",
            "d3 = reshape(take(pets.X, 0, 103), [1, 3, 64, 64])",
            "cats = concat(concat(concat(c0, c1, 0), c2, 0), c3, 0)",
            "dogs = concat(concat(concat(d0, d1, 0), d2, 0), d3, 0)",
            "X    = concat(cats, dogs, 0)",
            "ly0 = reshape(take(pets.Y, 0, 0),   [1])",
            "ly1 = reshape(take(pets.Y, 0, 1),   [1])",
            "ly2 = reshape(take(pets.Y, 0, 2),   [1])",
            "ly3 = reshape(take(pets.Y, 0, 3),   [1])",
            "ld0 = reshape(take(pets.Y, 0, 100), [1])",
            "ld1 = reshape(take(pets.Y, 0, 101), [1])",
            "ld2 = reshape(take(pets.Y, 0, 102), [1])",
            "ld3 = reshape(take(pets.Y, 0, 103), [1])",
            "ycats = concat(concat(concat(ly0, ly1, 0), ly2, 0), ly3, 0)",
            "ydogs = concat(concat(concat(ld0, ld1, 0), ld2, 0), ld3, 0)",
            "Y     = concat(ycats, ydogs, 0)",
            "# Model: four-head attention.",
            "linear_p   = linear(768, 128, 17)",
            "attn       = attention(128, 4, 23)                           # <-- 4 heads (was 1)",
            "classifier = chain(linear(128, 64, 31), relu_layer(), linear(64, 2, 37))",
            "# 30 full-batch adam steps. Same forward pipeline as the",
            "# single-head quick demo, just with heads=4.",
            "train 30 { adam(cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [128, 768])), [8, 16, 128])), 1, 0)), Y), [linear_p, attn, classifier], 0.001, 0.9, 0.999, 0.00000001); cross_entropy(apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [128, 768])), [8, 16, 128])), 1, 0)), Y) }",
            "loss_curve(last_losses)",
            "# Training accuracy on the 8-image set.",
            "logits   = apply(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(X, 16), [128, 768])), [8, 16, 128])), 1, 0))",
            "preds    = argmax(logits, 1)",
            "correct  = eq(preds, Y)",
            "accuracy = reduce_add(correct) / 8",
            "# Attention maps on the first (cat) image: pull image 0,",
            "# embed it, take batch slice 0, run attention_weights ->",
            "# [4, 16, 16] (heads x T_no_cls x T_no_cls).",
            "test_img    = reshape(take(pets.X, 0, 0), [1, 3, 64, 64])",
            "test_tokens = reshape(apply(linear_p, reshape(patchify(test_img, 16), [16, 768])), [16, 128])",
            "attn_maps   = attention_weights(attn, test_tokens)",
            "svg(attn_maps, \"heatmap_grid\")",
            "accuracy",
        ],
    },
];

#[cfg(test)]
mod tests {
    //! Web demo smoke. Walks every entry in `DEMOS`, lexes +
    //! parses + evals each line in fresh shared environment to
    //! catch syntax / runtime drift the moment a demo string in
    //! this file falls behind language changes. Mirrors what the
    //! "Run demo" button does in the browser, minus the
    //! visualization output.
    //!
    //! Skipped: REPL slash-commands (`:tags x`) and pure
    //! comments are not parseable as expressions; the
    //! browser handles those as side-channel commands. For
    //! long-running training demos we follow the same split as
    //! `all_demos_smoke`: a quick test exercises everything
    //! except the heavy ones, and a `#[ignore]`-gated test
    //! covers the heavies on demand.
    //!
    //! `PROGRESS_NOTES` invariant: every entry's `demo` matches
    //! a real `Demo::name` and `line_idx` is within that demo's
    //! `lines` length. A mismatch would silently fail to render
    //! the heads-up note in the browser.
    use super::{DEMOS, PROGRESS_NOTES};
    use mlpl_eval::{Environment, eval_program_value};
    use mlpl_parser::{lex, parse};
    use std::collections::HashSet;

    /// Demos that call external services or do heavy training.
    const SKIP_DEMOS: &[&str] = &[
        "LLM Tool Use",
        "MLX Remote Runner",
        "Tiny LM",
        "Tiny LM Generate",
        "Moons MLP",
        "Circles MLP",
        "Transformer Block",
        "Pets: cat vs dog (quick)",
        "Pets: predict + gallery",
        "Pets: multi-head ViT (quick + viz)",
    ];

    fn run_demo(demo_name: &str, lines: &[&str]) -> Result<(), String> {
        let mut env = Environment::new();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty()
                || trimmed.starts_with("//")
                || trimmed.starts_with('#')
                || (trimmed.starts_with(':') && !trimmed.starts_with("::"))
            {
                continue;
            }
            let toks = lex(line).map_err(|e| format!("[{demo_name} line {i}] lex: {e:?}"))?;
            let prog = parse(&toks).map_err(|e| format!("[{demo_name} line {i}] parse: {e:?}"))?;
            eval_program_value(&prog, &mut env)
                .map_err(|e| format!("[{demo_name} line {i}] eval: {e:?}"))?;
        }
        Ok(())
    }

    #[test]
    fn every_quick_web_demo_runs() {
        let mut failures: Vec<String> = Vec::new();
        for demo in DEMOS.iter() {
            if SKIP_DEMOS.contains(&demo.name) {
                continue;
            }
            if let Err(msg) = run_demo(demo.name, demo.lines) {
                failures.push(msg);
            }
        }
        assert!(
            failures.is_empty(),
            "{} web demo(s) regressed:\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }

    #[test]
    #[ignore = "heavy training demos take 30+s; run with --ignored"]
    fn every_heavy_web_demo_runs() {
        let mut failures: Vec<String> = Vec::new();
        for demo in DEMOS.iter() {
            if !SKIP_DEMOS.contains(&demo.name) {
                continue;
            }
            if matches!(demo.name, "LLM Tool Use" | "MLX Remote Runner") {
                continue;
            }
            if let Err(msg) = run_demo(demo.name, demo.lines) {
                failures.push(msg);
            }
        }
        assert!(
            failures.is_empty(),
            "{} heavy web demo(s) regressed:\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }

    #[test]
    fn progress_notes_reference_real_demo_lines() {
        let demos: HashSet<&str> = DEMOS.iter().map(|d| d.name).collect();
        let mut bad: Vec<String> = Vec::new();
        for note in PROGRESS_NOTES.iter() {
            if !demos.contains(note.demo) {
                bad.push(format!("unknown demo {:?}", note.demo));
                continue;
            }
            let demo = DEMOS.iter().find(|d| d.name == note.demo).unwrap();
            if note.line_idx >= demo.lines.len() {
                bad.push(format!(
                    "{}: line_idx {} >= lines.len() {}",
                    note.demo,
                    note.line_idx,
                    demo.lines.len()
                ));
            }
        }
        assert!(
            bad.is_empty(),
            "PROGRESS_NOTES drift:\n  - {}",
            bad.join("\n  - ")
        );
    }
}
