//! Saga 15 step 002: `lora(m, rank, alpha, seed)` builtin.
//!
//! Clone `m` and replace every `Linear` node in the tree with
//! a `LinearLora` that owns two fresh low-rank adapter
//! matrices alongside the cloned base `W`, `b`. Adapter `A`
//! is randn-initialized and scaled by `1 / sqrt(in_dim)`;
//! adapter `B` is zero-initialized so the pre-training-step
//! forward matches the base exactly. The cloned base `W` and
//! `b` of each `LinearLora` are auto-marked frozen so
//! `adam(loss, lora_m, ...)` only moves the adapters;
//! `unfreeze(lora_m)` opens the base to training as well.
//!
//! Forward + autograd for `LinearLora` is step 003. This
//! step ships the rewrite + param allocation only.
//!
//! See `contracts/eval-contract/lora.md`.

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::error::EvalError;
use crate::model::ModelSpec;
use crate::value::Value;

/// Shared rewrite context threaded through the `LinearLora`
/// tree walker. `counter` is mutated as each Linear is
/// wrapped so same-shape adapters get independent PRNG
/// seeds (`seed + counter`).
struct LoraCtx {
    rank: usize,
    alpha: f64,
    seed: f64,
    counter: usize,
}

/// Parameters for one `allocate_adapters` call.
struct AdapterInit<'a> {
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    device: &'a str,
    a_seed: f64,
}

/// `lora(m, rank, alpha, seed)` -- wrap every `Linear` in
/// `m` with trainable low-rank adapters.
pub(crate) fn eval_lora(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if args.len() != 4 {
        return Err(EvalError::BadArity {
            func: "lora".into(),
            expected: 4,
            got: args.len(),
        });
    }
    let source = resolve_model(&args[0], env)?;
    reject_nested_lora(&source)?;
    // `rank` must be a non-negative integer scalar.
    let rank_f = scalar_f64(&args[1], env)?;
    if !rank_f.is_finite() || rank_f < 0.0 || rank_f.fract() != 0.0 {
        return Err(EvalError::Unsupported(format!(
            "lora: rank must be a non-negative integer, got {rank_f}"
        )));
    }
    let rank = rank_f as usize;
    if rank == 0 {
        return Err(EvalError::Unsupported(
            "lora: rank must be positive, got 0".into(),
        ));
    }
    let alpha = scalar_f64(&args[2], env)?;
    let seed = scalar_f64(&args[3], env)?;
    // Clone the tree first so the base W, b belong to the
    // returned model (and mutating the clone's W via
    // `unfreeze(student)` then training does not touch the
    // caller's source model).
    let cloned = crate::model_clone::clone_spec(&source, env)?;
    let mut ctx = LoraCtx {
        rank,
        alpha,
        seed,
        counter: 0,
    };
    let student = rewrite_linears_to_lora(cloned, env, &mut ctx)?;
    // Auto-freeze every non-adapter param in the student
    // tree: embed tables, attention projections, rms_norm
    // (parameter-free; no-op), and the cloned base W, b of
    // each LinearLora. Only the newly allocated `__lora_A_*`
    // and `__lora_B_*` names stay trainable. This matches
    // the LoRA-library convention -- "frozen base +
    // trainable adapters" -- without the user having to
    // remember which params belong to the base.
    for name in student.params() {
        if !name.starts_with("__lora_A_") && !name.starts_with("__lora_B_") {
            env.mark_frozen(&name);
        }
    }
    Ok(student)
}

fn resolve_model(arg: &Expr, env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if let Expr::Ident(name, _) = arg {
        match env.get_model(name) {
            Some(m) => Ok(m.clone()),
            None => Err(EvalError::Unsupported(format!(
                "lora: '{name}' is not a model"
            ))),
        }
    } else {
        match crate::eval::eval_expr(arg, env, &mut None)? {
            Value::Model(m) => Ok(m),
            _ => Err(EvalError::Unsupported(
                "lora: first argument must evaluate to a model".into(),
            )),
        }
    }
}

fn reject_nested_lora(spec: &ModelSpec) -> Result<(), EvalError> {
    match spec {
        ModelSpec::LinearLora { .. } => Err(EvalError::Unsupported(
            "lora: model already has LoRA adapters; nested lora() is not supported".into(),
        )),
        ModelSpec::Chain(children) => {
            for c in children {
                reject_nested_lora(c)?;
            }
            Ok(())
        }
        ModelSpec::Residual(inner) => reject_nested_lora(inner),
        _ => Ok(()),
    }
}

fn rewrite_linears_to_lora(
    spec: ModelSpec,
    env: &mut Environment,
    ctx: &mut LoraCtx,
) -> Result<ModelSpec, EvalError> {
    match spec {
        ModelSpec::Linear { w, b } => wrap_linear(env, w, b, ctx),
        ModelSpec::Chain(children) => {
            let mut out = Vec::with_capacity(children.len());
            for c in children {
                out.push(rewrite_linears_to_lora(c, env, ctx)?);
            }
            Ok(ModelSpec::Chain(out))
        }
        ModelSpec::Residual(inner) => Ok(ModelSpec::Residual(Box::new(rewrite_linears_to_lora(
            *inner, env, ctx,
        )?))),
        ModelSpec::Activation(_)
        | ModelSpec::RmsNorm { .. }
        | ModelSpec::Embedding { .. }
        | ModelSpec::Attention { .. } => Ok(spec),
        ModelSpec::LinearLora { .. } => Err(EvalError::Unsupported(
            "lora: unexpected LinearLora in source tree (nested lora check should have caught this)"
                .into(),
        )),
    }
}

fn wrap_linear(
    env: &mut Environment,
    w: String,
    b: String,
    ctx: &mut LoraCtx,
) -> Result<ModelSpec, EvalError> {
    let w_arr = env
        .get(&w)
        .ok_or_else(|| EvalError::UndefinedVariable(w.clone()))?;
    let dims = w_arr.shape().dims();
    if dims.len() != 2 {
        return Err(EvalError::Unsupported(format!(
            "lora: base Linear W '{w}' must be rank-2, got rank {}",
            dims.len()
        )));
    }
    let (in_dim, out_dim) = (dims[0], dims[1]);
    let device = env.tensor_device(&w).to_string();
    let a_seed = ctx.seed + ctx.counter as f64;
    ctx.counter += 1;
    let (a_name, b_adapter_name) = allocate_adapters(
        env,
        &AdapterInit {
            in_dim,
            out_dim,
            rank: ctx.rank,
            device: &device,
            a_seed,
        },
    )?;
    // Auto-freeze of the cloned base W, b happens in
    // `eval_lora` after the whole tree is rewritten -- one
    // pass that freezes every non-adapter param so
    // embeddings, attention, and bare linears all end up
    // frozen consistently.
    Ok(ModelSpec::LinearLora {
        w,
        b,
        a: a_name,
        b_adapter: b_adapter_name,
        in_dim,
        out_dim,
        rank: ctx.rank,
        alpha: ctx.alpha,
    })
}

/// Allocate the A and B adapter parameters for one
/// `LinearLora` node. Validates `rank <= min(in, out)` as
/// part of the call so `wrap_linear` stays short. Returns
/// the fresh (A name, B name). A is `randn(a_seed, [in,
/// rank]) * (1 / sqrt(in))`; B is `zeros([rank, out])`.
fn allocate_adapters(
    env: &mut Environment,
    init: &AdapterInit<'_>,
) -> Result<(String, String), EvalError> {
    if init.rank > init.in_dim.min(init.out_dim) {
        return Err(EvalError::Unsupported(format!(
            "lora: rank {} exceeds min(in={}, out={}) for this Linear",
            init.rank, init.in_dim, init.out_dim
        )));
    }
    let id = env.next_model_id;
    env.next_model_id += 1;
    let a_name = format!("__lora_A_{id}");
    let b_adapter_name = format!("__lora_B_{id}");
    let shape_arr = DenseArray::new(
        Shape::new(vec![2]),
        vec![init.in_dim as f64, init.rank as f64],
    )?;
    let a_raw = mlpl_runtime::call_builtin(
        "randn",
        vec![DenseArray::from_scalar(init.a_seed), shape_arr],
    )?;
    let scale = 1.0 / (init.in_dim as f64).sqrt();
    let a_data: Vec<f64> = a_raw.data().iter().map(|v| v * scale).collect();
    let a_arr = DenseArray::new(Shape::new(vec![init.in_dim, init.rank]), a_data)?;
    env.set_param(a_name.clone(), a_arr);
    env.set_tensor_device(a_name.clone(), init.device.to_string());
    let b_arr = DenseArray::zeros(Shape::new(vec![init.rank, init.out_dim]));
    env.set_param(b_adapter_name.clone(), b_arr);
    env.set_tensor_device(b_adapter_name.clone(), init.device.to_string());
    Ok((a_name, b_adapter_name))
}

fn scalar_f64(expr: &Expr, env: &mut Environment) -> Result<f64, EvalError> {
    let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(
            "lora: rank, alpha, and seed must be scalars".into(),
        ));
    }
    Ok(arr.data()[0])
}
