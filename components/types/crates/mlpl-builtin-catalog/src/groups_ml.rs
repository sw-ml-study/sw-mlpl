//! ML-domain catalog groups.

use crate::FnGroup;

/// ML, dataset, model-DSL, visualization, and Engram groups.
pub(crate) const GROUPS: &[FnGroup] = &[
    (
        "ML primitives",
        &[
            ("argmax", "argmax(a[, axis])", "flat or per-axis argmax"),
            ("softmax", "softmax(a, axis)", "numerically stable softmax"),
            ("one_hot", "one_hot(labels, k)", "NxK one-hot encoding"),
            ("random", "random(seed, shape)", "seeded uniform [0, 1)"),
            ("randn", "randn(seed, shape)", "seeded standard normal"),
            (
                "blobs",
                "blobs(seed, n, centers)",
                "Nx3 gaussian-blob dataset",
            ),
            (
                "moons",
                "moons(seed, n, noise)",
                "two-moons synthetic dataset",
            ),
            (
                "circles",
                "circles(seed, n, noise)",
                "concentric-circles dataset",
            ),
            (
                "cross_entropy",
                "cross_entropy(logits, targets)",
                "scalar mean negative log-likelihood",
            ),
            (
                "perplexity",
                "perplexity(logits, targets)",
                "exp(cross_entropy(...)) -- the canonical LM metric",
            ),
            (
                "sinusoidal_encoding",
                "sinusoidal_encoding(seq_len, d_model)",
                "sinusoidal positional table",
            ),
            (
                "sample",
                "sample(logits, temperature, seed)",
                "categorical sample from a logit vector",
            ),
            (
                "top_k",
                "top_k(logits, k)",
                "mask all but top-k logits to -inf",
            ),
        ],
    ),
    (
        "Dataset prep",
        &[
            (
                "shuffle",
                "shuffle(x, seed)",
                "Fisher-Yates row permutation",
            ),
            (
                "batch",
                "batch(x, size)",
                "row batches with zero-padded tail",
            ),
            (
                "batch_mask",
                "batch_mask(x, size)",
                "0/1 mask matching batch(x, size)",
            ),
            ("split", "split(x, frac, seed)", "deterministic train chunk"),
            (
                "val_split",
                "val_split(x, frac, seed)",
                "complementary val chunk",
            ),
            (
                "shift_pairs_x",
                "shift_pairs_x(ids, block)",
                "next-token input windows",
            ),
            (
                "shift_pairs_y",
                "shift_pairs_y(ids, block)",
                "matching target windows for shift_pairs_x",
            ),
        ],
    ),
    (
        "Embeddings + manifold",
        &[
            (
                "pairwise_sqdist",
                "pairwise_sqdist(X)",
                "[N,N] squared Euclidean distances",
            ),
            ("knn", "knn(X, k)", "[N,k] nearest-neighbor indices"),
            (
                "knn_graph",
                "knn_graph(X, k)",
                "[N*k, 3] (i, j, dist) edge list -- UMAP's input layer",
            ),
            ("pca", "pca(X, k)", "top-k principal-component projection"),
            (
                "pca_components",
                "pca_components(X, k)",
                "[k,D] loadings: row i is the i-th PC direction",
            ),
            (
                "pca_variance_explained",
                "pca_variance_explained(X, k)",
                "[k] per-component variance-explained ratios",
            ),
            ("tsne", "tsne(X, perp, iters, seed)", "t-SNE 2D embedding"),
            (
                "umap",
                "umap(X, n_neighbors, min_dist, iters, seed)",
                "UMAP 2D embedding -- preserves both local and global structure",
            ),
            (
                "mds",
                "mds(X, k, iters, seed)",
                "Multidimensional Scaling -- preserves pairwise distances",
            ),
            (
                "random_projection",
                "random_projection(X, k, seed)",
                "Johnson-Lindenstrauss random projection (sanity baseline)",
            ),
        ],
    ),
    (
        "CNN + RNN",
        &[
            (
                "conv2d",
                "conv2d(input, filters, stride, padding)",
                "2D convolution: [B,C_in,H,W] x [C_out,C_in,kH,kW]",
            ),
            (
                "pool2d",
                "pool2d(input, size, mode)",
                "2D pooling: mode=1 max, mode=0 avg",
            ),
            (
                "rnn_cell",
                "rnn_cell(input, hidden, W_ih, W_hh, bias)",
                "one Elman RNN step: tanh(W_ih@input + W_hh@hidden + bias)",
            ),
            (
                "lstm_cell",
                "lstm_cell(input, hidden, cell, W, bias)",
                "one LSTM step: returns [hidden; cell] concat",
            ),
        ],
    ),
    (
        "Autograd + optimizers",
        &[
            ("grad", "grad(expr, wrt)", "reverse-mode gradient"),
            (
                "momentum_sgd",
                "momentum_sgd(loss, params, lr, beta)",
                "momentum-SGD update",
            ),
            ("adam", "adam(loss, params, lr, b1, b2, eps)", "Adam update"),
            (
                "cosine_schedule",
                "cosine_schedule(step, total, lr_min, lr_max)",
                "cosine LR schedule",
            ),
            (
                "linear_warmup",
                "linear_warmup(step, warmup, lr)",
                "linear warmup helper",
            ),
        ],
    ),
    (
        "Model DSL",
        &[
            ("linear", "linear(in, out, seed)", "dense layer y = xW + b"),
            ("chain", "chain(a, b, ...)", "sequential composition"),
            ("tanh_layer", "tanh_layer()", "tanh activation layer"),
            ("relu_layer", "relu_layer()", "relu activation layer"),
            (
                "softmax_layer",
                "softmax_layer()",
                "softmax activation layer",
            ),
            ("residual", "residual(inner)", "y = x + inner(x)"),
            ("rms_norm", "rms_norm(dim)", "per-row RMS normalization"),
            (
                "attention",
                "attention(d_model, heads, seed)",
                "multi-head self-attention",
            ),
            (
                "causal_attention",
                "causal_attention(d_model, heads, seed)",
                "self-attention with a lower-triangular causal mask",
            ),
            ("apply", "apply(model, X)", "forward pass on a stored model"),
            (
                "predict_batch",
                "predict_batch(model, X)",
                "forward pass + argmax over the trailing axis (integer class labels)",
            ),
        ],
    ),
    (
        "Visualization",
        &[
            ("svg", "svg(data, type[, aux])", "render an SVG diagram"),
            ("hist", "hist(values, bins)", "histogram"),
            (
                "scatter_labeled",
                "scatter_labeled(points, labels)",
                "colored scatter",
            ),
            ("loss_curve", "loss_curve(losses)", "training loss curve"),
            (
                "train_val_curve",
                "train_val_curve(train, val)",
                "train vs validation loss (overfitting gap)",
            ),
            (
                "loss_landscape",
                "loss_landscape(surface, dims, path)",
                "2-weight loss surface + optimizer trajectory",
            ),
            (
                "confusion_matrix",
                "confusion_matrix(pred, truth)",
                "KxK heatmap",
            ),
            (
                "boundary_2d",
                "boundary_2d(surface, grid, X, y)",
                "classifier boundary",
            ),
        ],
    ),
    (
        "Engram",
        &[
            (
                "engram",
                "engram(hidden, ngrams, heads, slots, head_dim, seed)",
                "conditional n-gram memory layer: zero table + near-closed concat gate (train to use)",
            ),
            (
                "apply_engram",
                "apply_engram(e, h, ids)",
                "engram forward: hash ids, gather memory, project, gate, add to the residual",
            ),
            (
                "engram_stats",
                "engram_stats(e, ids) or engram_stats(e, h, ids)",
                "addressing/memory/gate health record: rows_addressed, unique_rows, collisions, nonzero_rows, max_row_norm (+ gate_mean/gate_max with h)",
            ),
            (
                "ngram_hash",
                "ngram_hash(ids, orders, heads, slots, seed)",
                "rolling n-gram hash indices [T, order, head] (exact cross-backend contract)",
            ),
            (
                "gather_rows",
                "gather_rows(table, indices)",
                "select rows of a rank-2 table; output shape = indices shape + [dim]",
            ),
        ],
    ),
];
