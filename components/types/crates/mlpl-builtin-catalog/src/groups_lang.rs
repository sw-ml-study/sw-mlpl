//! Language/array-domain catalog groups.

use crate::FnGroup;

/// Array, linear algebra, math, and statistics groups.
pub(crate) const GROUPS: &[FnGroup] = &[
    (
        "Array",
        &[
            ("range", "range(n)", "integers 0..n as a vector (preferred)"),
            ("shape", "shape(a)", "dimension vector of a"),
            (
                "labels",
                "labels(a)",
                "comma-joined axis labels of a (empty for positional)",
            ),
            ("rank", "rank(a)", "number of dimensions of a"),
            (
                "depth",
                "depth(a)",
                "nesting depth: 0 for a scalar, 1 for a flat array",
            ),
            (
                "disp",
                "disp(a)",
                "ASCII box diagram of a showing its rank, shape, and depth",
            ),
            ("size", "size(a)", "total element count of a (numel)"),
            (
                "tally",
                "tally(a)",
                "length of the leading axis of a (major-cell count)",
            ),
            (
                "flatten",
                "flatten(a)",
                "ravel: all elements as a rank-1 vector in row-major order",
            ),
            ("reshape", "reshape(a, dims)", "reshape a to the given dims"),
            ("transpose", "transpose(a)", "reverse axis order"),
            (
                "get_value",
                "get_value(r)",
                "Ok side of a Result as a 0-or-1 element vector ([] when Err); tally is is_some",
            ),
            (
                "get_error",
                "get_error(r)",
                "Err side of a Result as a 0-or-1 element vector ([] when Ok)",
            ),
            (
                "emit_frame",
                "emit_frame(name, step, x)",
                "stream tensor x as a live frame (connect mode); no-op locally; returns x",
            ),
            (
                "rotate",
                "rotate(x, k, axis)",
                "cyclic shift along axis; negative k (spell it 0 - k) rotates the other way",
            ),
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
            (
                "running_product",
                "running_product(v)",
                "running product along a rank-1 vector",
            ),
            (
                "running_sum",
                "running_sum(v)",
                "running sum along a rank-1 vector",
            ),
            (
                "grade_up",
                "grade_up(v)",
                "stable argsort indices, ascending",
            ),
            (
                "grade_down",
                "grade_down(v)",
                "stable argsort indices, descending",
            ),
            (
                "compress",
                "compress(mask, a[, axis])",
                "keep slices where the mask is nonzero",
            ),
            (
                "rand_ints",
                "rand_ints(n, lo, hi, seed)",
                "n uniform ints in [lo, hi), seeded",
            ),
            (
                "dedupe_rows",
                "dedupe_rows(X)",
                "unique rows as {rows, index}",
            ),
            (
                "kg_neighbors",
                "kg_neighbors(edges, node[, rel])",
                "sorted unique one-hop ids",
            ),
            (
                "kg_verify",
                "kg_verify(edges, paths)",
                "per-row path-validity mask",
            ),
            (
                "kg_paths",
                "kg_paths(edges, hops, n, seed)",
                "sampled valid paths [n, hops+1]",
            ),
            (
                "kg_split",
                "kg_split(edges, frac, seed)",
                "entity-disjoint {seen, unseen} split",
            ),
            (
                "linspace",
                "linspace(start, stop, n)",
                "n evenly-spaced values from start to stop (inclusive)",
            ),
            ("zeros", "zeros(shape)", "array of zeros"),
            ("ones", "ones(shape)", "array of ones"),
            ("fill", "fill(shape, value)", "array filled with value"),
            ("grid", "grid(bounds, n)", "n*n by 2 (x,y) grid"),
            (
                "concat",
                "concat(a, b[, axis])",
                "concat rank-0/1 (2-arg) or axis-aware concat for tape (3-arg)",
            ),
            (
                "last_row",
                "last_row(M)",
                "final row of a rank-2 matrix as a vector",
            ),
            (
                "patchify",
                "patchify(x, P)",
                "[B,C,H,W] image batch into [B,N,P*P*C] patches",
            ),
            (
                "take",
                "take(x, axis, idx)",
                "drop one axis at a single integer index (tape-differentiable)",
            ),
            (
                "argtop_k",
                "argtop_k(scores, k)",
                "indices of top-k entries (descending)",
            ),
            (
                "scatter",
                "scatter(buf, idx, vals)",
                "copy of rank-1 buf with buf[idx] replaced by val (single cell)",
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
            ("pi", "pi()", "3.14159265... (zero-arg constant)"),
            ("e", "e()", "2.71828182... (zero-arg constant)"),
            ("exp", "exp(a)", "elementwise exponential"),
            ("log", "log(a)", "elementwise natural log"),
            ("sqrt", "sqrt(a)", "elementwise square root"),
            ("abs", "abs(a)", "elementwise absolute value"),
            ("sin", "sin(a)", "elementwise sine (radians)"),
            ("cos", "cos(a)", "elementwise cosine (radians)"),
            ("floor", "floor(a)", "elementwise floor"),
            ("ceil", "ceil(a)", "elementwise ceiling"),
            ("round", "round(a)", "elementwise round to nearest integer"),
            ("pow", "pow(a, b)", "elementwise power"),
            ("mod", "mod(a, b)", "elementwise remainder (a % b)"),
            ("relu", "relu(a)", "elementwise max(0, a)"),
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
];
