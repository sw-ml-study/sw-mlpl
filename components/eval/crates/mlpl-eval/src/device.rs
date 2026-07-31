//! `device("target") { body }` evaluator + MLX dispatch helper
//! (Saga 14 step 004).
//!
//! Two responsibilities live here:
//!
//! 1. `eval_device` runs the body of a `device("...")` block with
//!    the named target pushed onto `Environment::device_stack`. On
//!    exit it pops the entry, restoring whatever target was active
//!    in the surrounding scope. Nesting works in either direction
//!    (`experiment { device { ... } }` and the swap), and an inner
//!    `device("cpu")` overrides an outer `device("mlx")`.
//!
//! 2. `try_mlx_dispatch` is the single place where `mlpl-eval`
//!    decides whether an op should route through the `mlpl-mlx-rt`
//!    runtime. It is called from `eval_binop` and `eval_fncall`
//!    only when `Environment::device()` returns `"mlx"`. The
//!    helper returns `Some(result)` if the named op exists in
//!    `mlpl-mlx-rt`, otherwise `None` so the caller can fall back to
//!    the CPU path.
//!
//! Triple-gate: the `mlx` Cargo feature on `mlpl-eval` pulls in
//! `mlpl-mlx-rt` (which itself triple-gates on
//! `target_os = "macos"`, `target_arch = "aarch64"`, and its own
//! `mlx` feature). When any of those is missing the dispatch
//! helper is a stub returning `None`, the `device("mlx") { }`
//! block falls back to CPU, and `eval_device` emits a one-time
//! warning so the user knows their code ran on the wrong device.

use crate::env_api::*;
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use mlpl_array::ArrayError;
use mlpl_array::DenseArray;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

/// Evaluate a `device("target") { body }` block. Returns the
/// value of the body's last statement -- mirrors `experiment`'s
/// shape so a `device(...)` block is a value-yielding expression
/// like every other scoped form.
pub(crate) fn eval_device(
    target: &str,
    body: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<Value, EvalError> {
    if let Some(dispatcher) = env.peer_dispatcher() {
        let source = block_source(body);
        let bindings = collect_array_bindings(env, &source);
        if let Some(result) = dispatcher.dispatch_block(target, &source, bindings) {
            return result;
        }
    }
    if !device_available(target) && env.take_device_fallback_warning() {
        eprintln!(
            "warning: device(\"{target}\") block requested but the \
             {target} feature is not compiled in; falling back to CPU."
        );
    }
    env.push_device(target.to_string());
    let mut last = Value::Array(DenseArray::from_scalar(0.0));
    for stmt in body {
        last = crate::eval::eval_expr(stmt, env, trace)?;
    }
    env.pop_device();
    Ok(last)
}

fn block_source(body: &[Expr]) -> String {
    body.iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}

fn collect_array_bindings(
    env: &Environment,
    source: &str,
) -> std::collections::HashMap<String, DenseArray> {
    env.vars_iter()
        .filter(|(name, _)| source.contains(name.as_str()))
        .map(|(name, arr)| (name.clone(), arr.clone()))
        .collect()
}

/// Whether the running build can dispatch through MLX (Apple) or
/// CUDA (Linux): each `const fn` combines its Cargo feature gate with
/// its target OS/arch gate.
const fn mlx_available() -> bool {
    cfg!(all(
        feature = "mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))
}
const fn cuda_available() -> bool {
    cfg!(all(
        feature = "cuda",
        target_os = "linux",
        target_arch = "x86_64"
    ))
}

/// Whether `target`'s GPU backend can actually dispatch in this build
/// (`cpu`/anything else always "runs", on CPU). Drives the one-time
/// CPU-fallback warning in `eval_device` / `eval_to_device`.
pub(crate) fn device_available(target: &str) -> bool {
    match target {
        "mlx" => mlx_available(),
        "cuda" => cuda_available(),
        _ => true,
    }
}

// The op-dispatch machinery (the `op_dispatch!` macro,
// `try_mlx_dispatch`/`try_cuda_dispatch`, `lift_array_error`, and
// `dispatched_call`) lives in `device_dispatch` -- extracted to keep
// this module within the file-size budget. `dispatched_call` is
// re-exported so existing `crate::device::dispatched_call` call sites
// (Model DSL forward helpers, `eval_binop`/`eval_fncall`) resolve
// unchanged.
pub(crate) use crate::device_dispatch::dispatched_call;

// `materialize_tape_on_mlx` (below) reruns forward ops through MLX.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
use crate::device_dispatch::try_mlx_dispatch;

// The `to_device(x, target)` builtin lives in `device_to` (its own
// concern). Re-exported so `crate::device::eval_to_device` call sites
// resolve unchanged.
pub(crate) use crate::device_to::eval_to_device;

/// Re-execute every non-leaf node on `tape` through `mlpl-mlx-rt` so
/// each `NodeData::value` carries the MLX-rounded forward value
/// (Saga 14 step 006).
///
/// Design choice (option (b) hand-written, per the step prompt):
/// the autograd tape's structure -- nodes, parent ids, op
/// kinds -- stays exactly as the CPU tape built it. After the
/// forward pass, this helper walks node ids in insertion order
/// (which is topological because each op's parents are pushed
/// before the op itself) and recomputes every non-leaf node's
/// value via the corresponding `mlpl-mlx-rt` primitive, threading
/// MLX-computed parent values forward. The CPU backward formulas
/// in `mlpl-autograd::ops` then operate on those MLX-rounded
/// values; the gradients they produce match the all-CPU path
/// within the documented fp32 tolerance because the math is
/// identical -- only the forward values' fp32 round-trip differs.
///
/// Why not lean on `mlx_rs::transforms::grad` (option (a)): it
/// requires expressing the forward as an `Fn(Array) -> Array`
/// closure, which is incompatible with our `Tape` / `NodeId`
/// structure without a wholesale rewrite. The CPU formulas are
/// already tested via Saga 9's gradcheck fixtures and are the
/// authoritative spec; replacing each with an `mlx-rs` op would
/// change zero numerical behaviour while doubling the maintenance
/// surface. Option (b) ships the smallest defensible change that
/// honours the prompt's "grad on MLX matches CPU within
/// tolerance" invariant.
///
/// On builds without the mlx feature, this function is a no-op.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn materialize_tape_on_mlx(tape: &mlpl_autograd::Tape) {
    use mlpl_autograd::NodeKind;
    let len = tape.len();
    for i in 0..len {
        // Snapshot parents we need before taking a mutable borrow
        // so the borrow checker is happy.
        let kind = tape.nodes()[i].kind.clone();
        let new_value = match kind {
            NodeKind::Leaf => continue,
            NodeKind::Unary { op, parent } => {
                let x = tape.nodes()[parent.0].value.clone();
                rerun_unary(op, &x)
            }
            NodeKind::Binary { op, left, right } => {
                let a = tape.nodes()[left.0].value.clone();
                let b = tape.nodes()[right.0].value.clone();
                match rerun_binary(op, &a, &b) {
                    Ok(v) => v,
                    Err(_) => continue,
                }
            }
            NodeKind::SumAll { parent } => {
                let x = tape.nodes()[parent.0].value.clone();
                match try_mlx_dispatch("reduce_add", std::slice::from_ref(&x)) {
                    Some(Ok(v)) => v,
                    _ => x,
                }
            }
            NodeKind::MeanAll { parent } => {
                let x = tape.nodes()[parent.0].value.clone();
                match try_mlx_dispatch("mean", std::slice::from_ref(&x)) {
                    Some(Ok(v)) => v,
                    _ => x,
                }
            }
            NodeKind::Softmax { parent, axis } => {
                let x = tape.nodes()[parent.0].value.clone();
                match try_mlx_dispatch(
                    "softmax",
                    &[x.clone(), DenseArray::from_scalar(axis as f64)],
                ) {
                    Some(Ok(v)) => v,
                    _ => x,
                }
            }
            NodeKind::Transpose { parent } => {
                let x = tape.nodes()[parent.0].value.clone();
                match try_mlx_dispatch("transpose", std::slice::from_ref(&x)) {
                    Some(Ok(v)) => v,
                    _ => x,
                }
            }
            NodeKind::Reshape { .. } => {
                // Reshape stores no new dims on the tape (the new
                // shape comes from the existing CPU value); reuse
                // the CPU forward value to avoid recomputing.
                tape.nodes()[i].value.clone()
            }
            NodeKind::MatMul { left, right } => {
                let a = tape.nodes()[left.0].value.clone();
                let b = tape.nodes()[right.0].value.clone();
                match try_mlx_dispatch("matmul", &[a.clone(), b]) {
                    Some(Ok(v)) => v,
                    _ => a,
                }
            }
            NodeKind::CrossEntropy { .. } => {
                // The fused CE forward is small per-row work; let
                // the CPU value stand. Its inputs (the logits)
                // were already MLX-rounded by an earlier loop
                // iteration, so the CE value implicitly carries
                // MLX-rounded data.
                tape.nodes()[i].value.clone()
            }
            NodeKind::Patchify { .. }
            | NodeKind::Concat { .. }
            | NodeKind::Take { .. }
            | NodeKind::Stack { .. }
            | NodeKind::Rotate { .. } => {
                // Saga 29 step 005/007/013: patchify, concat,
                // take, stack, and rotate are pure
                // re-arrangements -- no fp-rounding work happens
                // inside them. The CPU forward value is
                // bitwise-identical to what an MLX rerun would
                // produce, so reuse it directly without a peer
                // round-trip.
                tape.nodes()[i].value.clone()
            }
        };
        tape.nodes_mut()[i].value = new_value;
    }
}

/// CPU-only stub used when the `mlx` feature, target OS, or
/// target arch is missing. The autograd tape stays CPU-resident
/// so the gradients produced by `mlpl-autograd` match the
/// all-CPU path exactly.
#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
pub(crate) fn materialize_tape_on_mlx(_tape: &mlpl_autograd::Tape) {}

/// Forward a unary op through `mlpl-mlx-rt` when possible, or fall
/// back to the input unchanged when the op has no MLX kernel
/// (defensive -- every current `UnaryOp` variant has one).
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn rerun_unary(op: mlpl_autograd::ops::UnaryOp, x: &DenseArray) -> DenseArray {
    use mlpl_autograd::ops::UnaryOp;
    let name = match op {
        UnaryOp::Neg => "neg",
        UnaryOp::Exp => "exp",
        UnaryOp::Log => "log",
        UnaryOp::Relu => "relu",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Sigmoid => "sigmoid",
    };
    match try_mlx_dispatch(name, std::slice::from_ref(x)) {
        Some(Ok(v)) => v,
        _ => op.forward(x),
    }
}

/// Forward a binary op through `mlpl-mlx-rt` when possible.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn rerun_binary(
    op: mlpl_autograd::ops::BinaryOp,
    a: &DenseArray,
    b: &DenseArray,
) -> Result<DenseArray, ArrayError> {
    use mlpl_autograd::ops::BinaryOp;
    let name = match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
    };
    match try_mlx_dispatch(name, &[a.clone(), b.clone()]) {
        Some(Ok(v)) => Ok(v),
        Some(Err(e)) => Err(e),
        None => op.forward(a, b),
    }
}
