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
                "transpose_axes",
                "transpose_axes(a, perm)",
                "generalized dyadic transpose: result axis i = source axis perm[i] (0-based)",
            ),
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
                "call",
                "call(f, args...)",
                "invoke a function reference (:u:name or :name) uniformly",
            ),
            (
                "map_ok",
                "map_ok(f, r)",
                "apply f inside ok(...); err passes through",
            ),
            (
                "and_then",
                "and_then(f, r)",
                "chain: ok(x) -> f(x) returning a Result; err passes through",
            ),
            (
                "or_else",
                "or_else(f, r)",
                "recover: err(e) -> f(e); ok passes through",
            ),
            (
                "bracket",
                "bracket(setup, use, teardown)",
                "guaranteed-finally: teardown always runs after setup succeeds",
            ),
            (
                "tests",
                "tests()",
                "stable names of @test-annotated functions, in source order",
            ),
            (
                "test_info",
                "test_info(name)",
                "one test's registry row: name/fn/tags/skip/expected_failure/timeout_ms/source/line",
            ),
            (
                "annotations",
                "annotations(name)",
                "a definition's @word annotations as a record; bare words map to 1",
            ),
            (
                "test_event_sink",
                "test_event_sink(f)",
                "register the :u: callback that receives emitted test events",
            ),
            (
                "emit_test_event",
                "emit_test_event(e)",
                "validate a v1 test-event record, deliver to the sink and host channel",
            ),
            (
                "expunge",
                "expunge(name)",
                "free a binding or u: function (APL quad-EX); 1 = free, 0 = malformed",
            ),
            (
                "global_set",
                "global_set(name, v)",
                "explicit global write that survives the function frame (reporter state)",
            ),
            (
                "fs_walk",
                "fs_walk(root, opts)",
                "sandboxed lexical directory walk; opts: recursive/kind/pattern",
            ),
            (
                "read_text",
                "read_text(path)",
                "sandboxed exact-text file read; returns ok(text)/err",
            ),
            (
                "write_text",
                "write_text(path, s)",
                "sandboxed exact-text file write; returns ok(1)/err",
            ),
            (
                "read_bytes",
                "read_bytes(path)",
                "sandboxed raw-byte file read; returns ok(rank-1 array 0..256)/err",
            ),
            (
                "write_bytes",
                "write_bytes(path, bytes)",
                "sandboxed raw-byte file write; bytes a rank-<=1 array of integers 0..=255; returns ok(1)/err",
            ),
            (
                "write_atomic",
                "write_atomic(path, value)",
                "sandboxed crash-safe write (temp file + rename); value a string or byte array; returns ok(1)/err",
            ),
            (
                "remove_path",
                "remove_path(path)",
                "sandboxed file/dir removal; returns ok(1)/err",
            ),
            (
                "each",
                "each(f, v)",
                "apply a function reference per element; shape preserved (APL2 each)",
            ),
            (
                "table",
                "table(f, a, b)",
                "outer product over f: [m] x [n] -> [m, n] (APL2 jot-dot, BQN table)",
            ),
            (
                "atop",
                "atop(f, g, x...)",
                "composition: f(g(x...)) (BQN atop)",
            ),
            (
                "over",
                "over(f, g, x, y)",
                "composition: f(g(x), g(y)) (BQN over)",
            ),
            (
                "parse_json",
                "parse_json(s[, opts])",
                "JSON text -> typed value; opts {max_depth, max_bytes, results} cap nesting/size, rebuild ok/err",
            ),
            (
                "to_toml",
                "to_toml(record)",
                "record -> deterministic TOML text (sorted keys, nested records as [sections]); ok(text)/err",
            ),
            (
                "parse_toml",
                "parse_toml(text[, opts])",
                "TOML config subset -> record; opts {max_depth, max_bytes, results} as parse_json; ok(record)/err",
            ),
            (
                "to_json",
                "to_json(v)",
                "value -> deterministic JSON string (encode half of parse_json)",
            ),
            (
                "run_script",
                "run_script(path, opts)",
                "execute a script in a FRESH environment; structured status + captured events",
            ),
            (
                "equal",
                "equal(a, b)",
                "total structural equality over any two values; never errors",
            ),
            (
                "repr",
                "repr(v)",
                "bounded deterministic rendering for diagnostics",
            ),
            (
                "type_of",
                "type_of(v)",
                "stable kind string of any value (array/string/record/result/...); total, never errors",
            ),
            (
                "pareto_front",
                "pareto_front(P, dirs)",
                "mask of non-dominated rows; dirs: 1 max / -1 min per column",
            ),
            (
                "rand_ints",
                "rand_ints(n, lo, hi, seed)",
                "n uniform ints in [lo, hi), seeded",
            ),
            (
                "clock_ms",
                "clock_ms()",
                "monotonic elapsed milliseconds for benchmarking (native/connect only)",
            ),
            (
                "band",
                "band(a, b)",
                "bitwise AND over non-negative integers (element-wise, broadcast)",
            ),
            ("bor", "bor(a, b)", "bitwise OR"),
            (
                "bxor",
                "bxor(a, b)",
                "bitwise XOR (Hamming distance = popcount(bxor(a, b)))",
            ),
            (
                "bnot",
                "bnot(x, width)",
                "bitwise complement within width bits",
            ),
            (
                "popcount",
                "popcount(x)",
                "number of set bits in each element",
            ),
            (
                "shl",
                "shl(x, k, width)",
                "fixed-width left shift: (x << k) masked to width bits",
            ),
            ("shr", "shr(x, k)", "logical right shift by k bits"),
            (
                "bmask",
                "bmask(x, width)",
                "keep the low width bits (truncate / width conversion)",
            ),
            (
                "bits",
                "bits(x, width)",
                "scalar to a [width] 0/1 vector, LSB-first",
            ),
            (
                "from_bits",
                "from_bits(v)",
                "pack a 0/1 vector to an integer (inverse of bits)",
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
