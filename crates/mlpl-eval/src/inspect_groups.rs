//! Curated `:builtins` / `:describe <name>` table.
//!
//! Owns only the static data that drives the REPL's grouped
//! function listing -- no formatting or dispatch logic. Lives in
//! its own module so `inspect.rs` (which holds the formatters and
//! the `:vars` / `:models` / `:wsid` paths) stays under the
//! sw-checklist file-LOC budget. The data here grows whenever a
//! new builtin lands; the formatters do not.
//!
//! The `tests/help_completeness_tests.rs` integration test asserts
//! that every name in this table appears in
//! `docs/lang-reference.md` and that every runtime dispatch arm
//! aggregated by `mlpl_runtime::runtime_builtin_names` is covered
//! here.

/// `(name, signature, one-line doc)` row used by both the grouped
/// `:fns` listing and the flat `:describe <builtin>` lookup.
pub(crate) type FnEntry = (&'static str, &'static str, &'static str);
/// `(group_label, entries)` used by `:fns`.
pub(crate) type FnGroup = (&'static str, &'static [FnEntry]);

pub(crate) const BUILTIN_GROUPS: &[FnGroup] = &[
    (
        "Array",
        &[
            ("iota", "iota(n)", "integers 0..n as a vector"),
            ("shape", "shape(a)", "dimension vector of a"),
            (
                "labels",
                "labels(a)",
                "comma-joined axis labels of a (empty for positional)",
            ),
            ("rank", "rank(a)", "number of dimensions of a"),
            ("reshape", "reshape(a, dims)", "reshape a to the given dims"),
            ("transpose", "transpose(a)", "reverse axis order"),
            (
                "reduce",
                "reduce(:op, a[, axis])",
                "higher-order reduction: :op is :add/:+, :mul/:*, :min, :max, :and, :or",
            ),
            (
                "reduce_add",
                "reduce_add(a[, axis])",
                "sum all or along axis (== reduce(:add, ...))",
            ),
            (
                "reduce_mul",
                "reduce_mul(a[, axis])",
                "product all or along axis (== reduce(:mul, ...))",
            ),
            ("zeros", "zeros(shape)", "array of zeros"),
            ("ones", "ones(shape)", "array of ones"),
            ("fill", "fill(shape, value)", "array filled with value"),
            ("grid", "grid(bounds, n)", "n*n by 2 (x,y) grid"),
            (
                "concat",
                "concat(a, b)",
                "concat two rank-0/1 arrays into a vector",
            ),
            (
                "last_row",
                "last_row(M)",
                "final row of a rank-2 matrix as a vector",
            ),
            (
                "argtop_k",
                "argtop_k(scores, k)",
                "indices of top-k entries (descending)",
            ),
            (
                "scatter",
                "scatter(buf, idx, vals)",
                "accumulate vals into buf at idx",
            ),
        ],
    ),
    (
        "Linear algebra",
        &[
            ("dot", "dot(a, b)", "vector dot product"),
            ("matmul", "matmul(a, b)", "matrix multiplication"),
        ],
    ),
    (
        "Math",
        &[
            ("exp", "exp(a)", "elementwise exponential"),
            ("log", "log(a)", "elementwise natural log"),
            ("sqrt", "sqrt(a)", "elementwise square root"),
            ("abs", "abs(a)", "elementwise absolute value"),
            ("pow", "pow(a, b)", "elementwise power"),
            ("sigmoid", "sigmoid(a)", "logistic sigmoid activation"),
            ("tanh_fn", "tanh_fn(a)", "hyperbolic tangent activation"),
        ],
    ),
    (
        "Comparisons + statistics",
        &[
            ("gt", "gt(a, b)", "elementwise greater-than (0/1)"),
            ("lt", "lt(a, b)", "elementwise less-than (0/1)"),
            ("eq", "eq(a, b)", "elementwise equality (0/1)"),
            ("mean", "mean(a)", "mean of all elements"),
        ],
    ),
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
            ("pca", "pca(X, k)", "top-k principal-component projection"),
            ("tsne", "tsne(X, perp, iters, seed)", "t-SNE 2D embedding"),
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
];

/// Iterate every builtin name listed in [`BUILTIN_GROUPS`]. Used
/// by `tests/help_completeness_tests.rs` to assert every entry is
/// also documented in `docs/lang-reference.md`. Order matches the
/// curated category layout shown by `:builtins`.
pub fn documented_builtin_names() -> impl Iterator<Item = &'static str> {
    BUILTIN_GROUPS
        .iter()
        .flat_map(|(_, fns)| fns.iter().map(|(name, _, _)| *name))
}
