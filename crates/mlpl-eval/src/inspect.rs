//! REPL introspection commands: `:vars`, `:models`, `:fns`, `:wsid`,
//! and `:describe <name>`. Shared between the terminal REPL
//! (`mlpl-repl`) and the web REPL (`mlpl-web` via `mlpl-wasm`) so
//! that both surfaces behave identically.
//!
//! These are inspired by APL's workspace conventions (`)VARS`,
//! `)FNS`, `)WSID`) but delivered as REPL commands rather than
//! language-level built-ins, so they stay out of the expression
//! grammar and never need to return a value.

use mlpl_array::DenseArray;
use mlpl_core::LabeledShape;

use crate::env::Environment;
use crate::model::{ActKind, ModelSpec};

/// If `input` is a recognized introspection command, returns the
/// rendered output. Returns `None` when the command is not one of
/// ours -- the caller should pass it through its normal handling
/// path (error for unknown commands, etc.).
pub fn inspect(env: &Environment, input: &str) -> Option<String> {
    let trimmed = input.trim();
    if !trimmed.starts_with(':') {
        return None;
    }
    let mut parts = trimmed.split_whitespace();
    let head = parts.next()?;
    let arg = parts.next();
    match head {
        ":vars" | ":variables" => Some(format_vars(env)),
        ":models" => Some(format_models(env)),
        ":fns" | ":functions" => Some(
            "(no user-defined functions)\n\
             user functions are not yet a language feature; for the \
             built-in surface, use :builtins"
                .into(),
        ),
        ":builtins" | ":built-ins" => Some(format_builtins()),
        ":experiments" => Some(crate::experiment::format_registry(env)),
        ":version" => Some(format!(
            "MLPL v{} -- Array Programming Language for ML\n  \
             target: {}",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::ARCH,
        )),
        ":wsid" => Some(format!(
            "workspace:\n  variables:       {}\n  parameters:      {}\n  \
             models:          {}\n  optimizer slots: {}",
            env.vars.len(),
            env.params.len(),
            env.models.len(),
            env.optim_state.buffers.len()
        )),
        ":describe" => Some(match arg {
            Some(name) => format_describe(env, name),
            None => "usage: :describe <name>".into(),
        }),
        ":help" => arg.and_then(|topic| help_topic(topic, env)),
        _ => None,
    }
}

/// Resolve `:help <topic>` to a corresponding inspector output.
/// `:help` with no topic is handled by the REPL itself (it prints
/// the long-form cheatsheet); only `:help <topic>` lands here.
fn help_topic(topic: &str, env: &Environment) -> Option<String> {
    match topic {
        "vars" | "variables" => Some(format_vars(env)),
        "models" => Some(format_models(env)),
        "fns" | "functions" => Some(
            "(no user-defined functions)\n\
             user functions are not yet a language feature; for the \
             built-in surface, use :builtins"
                .into(),
        ),
        "builtins" | "built-ins" => Some(format_builtins()),
        "wsid" | "workspace" => Some(format!(
            "workspace:\n  variables:       {}\n  parameters:      {}\n  \
             models:          {}\n  optimizer slots: {}",
            env.vars.len(),
            env.params.len(),
            env.models.len(),
            env.optim_state.buffers.len()
        )),
        "describe" => Some(
            ":describe <name>\n  print the shape and a values preview \
             for a variable, the layer tree for a model, or the signature \
             and one-line doc for a built-in"
                .into(),
        ),
        _ => None,
    }
}

fn format_vars(env: &Environment) -> String {
    if env.vars.is_empty() {
        return "(no variables bound)".into();
    }
    let mut names: Vec<&String> = env.vars.keys().collect();
    names.sort();
    let mut out = String::new();
    for name in names {
        let arr = &env.vars[name];
        let shape = format_shape(arr);
        let tag = if env.params.contains(name) {
            " [param]"
        } else {
            ""
        };
        out.push_str(&format!("  {name}: {shape}{tag}\n"));
    }
    out.truncate(out.trim_end().len());
    out
}

fn format_models(env: &Environment) -> String {
    if env.models.is_empty() {
        return "(no models bound)".into();
    }
    let mut names: Vec<&String> = env.models.keys().collect();
    names.sort();
    let mut out = String::new();
    for name in names {
        let spec = &env.models[name];
        let param_count = spec.params().len();
        out.push_str(&format!(
            "  {name}: {} ({param_count} params)\n",
            render_spec(spec)
        ));
    }
    out.truncate(out.trim_end().len());
    out
}

fn format_describe(env: &Environment, name: &str) -> String {
    if let Some(tok) = env.tokenizers.get(name) {
        return format!("{name} -- tokenizer\n  {}", tok.describe());
    }
    if let Some(spec) = env.models.get(name) {
        let mut out = format!("{name} -- model\n  shape: {}\n", render_spec(spec));
        let ps = spec.params();
        if ps.is_empty() {
            out.push_str("  params: (none)");
        } else {
            out.push_str("  params:\n");
            for p in ps {
                if let Some(arr) = env.vars.get(&p) {
                    out.push_str(&format!("    {p}: {}\n", format_shape(arr)));
                }
            }
            out.truncate(out.trim_end().len());
        }
        return out;
    }
    if let Some(arr) = env.vars.get(name) {
        let shape = format_shape(arr);
        let tag = if env.params.contains(name) {
            " (trainable param)"
        } else {
            ""
        };
        let data = arr.data();
        let preview = if data.is_empty() {
            "(empty)".to_string()
        } else {
            let take = 8.min(data.len());
            let head: Vec<String> = data[..take].iter().map(|v| format!("{v:.4}")).collect();
            if data.len() > take {
                format!("{} ... ({} total)", head.join(" "), data.len())
            } else {
                head.join(" ")
            }
        };
        return format!("{name} -- array\n  shape: {shape}{tag}\n  values: {preview}");
    }
    if let Some(s) = env.get_string(name) {
        // Web-UI demos bind `_demo` here; multi-line indented.
        let body = s
            .lines()
            .map(|l| format!("  {l}"))
            .collect::<Vec<_>>()
            .join("\n");
        return format!("{name} -- string ({} chars)\n{body}", s.len());
    }
    // Last fallback: flatten the grouped built-in list and look up by name.
    for (_, entries) in BUILTIN_GROUPS {
        if let Some(doc) = entries.iter().find(|(n, _, _)| *n == name) {
            return format!("{} -- built-in\n  {}\n  {}", doc.0, doc.1, doc.2);
        }
    }
    format!("'{name}' is not a bound variable, model, or built-in.")
}

/// `(name, signature, one-line doc)` row used by both the grouped
/// `:fns` listing and the flat `:describe <builtin>` lookup.
type FnEntry = (&'static str, &'static str, &'static str);
/// `(group_label, entries)` used by `:fns`.
type FnGroup = (&'static str, &'static [FnEntry]);

const BUILTIN_GROUPS: &[FnGroup] = &[
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
                "reduce_add",
                "reduce_add(a[, axis])",
                "sum all or along axis",
            ),
            (
                "reduce_mul",
                "reduce_mul(a[, axis])",
                "product all or along axis",
            ),
            ("zeros", "zeros(shape)", "array of zeros"),
            ("ones", "ones(shape)", "array of ones"),
            ("fill", "fill(shape, value)", "array filled with value"),
            ("grid", "grid(bounds, n)", "n*n by 2 (x,y) grid"),
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

fn format_builtins() -> String {
    let mut out = String::new();
    for (group, fns) in BUILTIN_GROUPS {
        out.push_str(group);
        out.push('\n');
        for (_, sig, doc) in *fns {
            out.push_str(&format!("  {sig:<40} {doc}\n"));
        }
        out.push('\n');
    }
    out.truncate(out.trim_end().len());
    out
}

fn format_shape(arr: &DenseArray) -> String {
    let dims = arr.shape().dims();
    if dims.is_empty() {
        return "scalar".into();
    }
    // Labeled (fully or partially) arrays render through `LabeledShape`
    // Display: `[seq=6, d_model=4]`, `[6, d_model=4]`. Unlabeled
    // arrays keep the positional `[6, 4]` rendering so existing
    // :vars/:describe output is unchanged for pre-labels demos.
    if let Some(labels) = arr.labels() {
        return LabeledShape::new(dims.to_vec(), labels.to_vec()).to_string();
    }
    let inner: Vec<String> = dims.iter().map(usize::to_string).collect();
    format!("[{}]", inner.join(", "))
}

fn render_spec(spec: &ModelSpec) -> String {
    match spec {
        ModelSpec::Linear { .. } => "linear".into(),
        ModelSpec::Chain(children) => {
            let parts: Vec<String> = children.iter().map(render_spec).collect();
            format!("chain({})", parts.join(" -> "))
        }
        ModelSpec::Activation(k) => match k {
            ActKind::Tanh => "tanh".into(),
            ActKind::Relu => "relu".into(),
            ActKind::Softmax => "softmax".into(),
        },
        ModelSpec::Residual(inner) => format!("residual({})", render_spec(inner)),
        ModelSpec::RmsNorm { dim } => format!("rms_norm({dim})"),
        ModelSpec::Attention {
            d_model,
            heads,
            causal,
            ..
        } => {
            let name = if *causal {
                "causal_attention"
            } else {
                "attention"
            };
            format!("{name}(d={d_model}, heads={heads})")
        }
        ModelSpec::Embedding { vocab, d_model, .. } => {
            format!("embed[vocab={vocab}, d={d_model}]")
        }
    }
}
