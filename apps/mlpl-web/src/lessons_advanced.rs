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
