//! Per-step optimizer updates: `momentum_sgd` and `adam`.
//!
//! Split out of `grad.rs` per docs/code_metrics.md (file LOC
//! 555 over the 500 budget). The two optimizers share a
//! parameter-walking shape, both call `crate::grad::eval_grad`
//! to compute the gradient per tracked param, and both store
//! per-call buffers on `Environment::optim_state`. Keeping them
//! here keeps grad.rs focused on the autograd tape itself.

use crate::env_api::*;
use mlpl_array::DenseArray;
use mlpl_core::Span;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_types::EvalError;
use mlpl_eval_types::Value;

/// FnCall-dispatch wrapper around `momentum_sgd` / `adam`. Lifted
/// out of `eval::eval_expr` for saga 33 step 023 so eval.rs drops
/// below its file LOC budget. Emits a single trace event labelled
/// with the optimizer name.
pub(crate) fn eval_optim(
    name: &str,
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
    span: &Span,
) -> Result<Value, EvalError> {
    if let Some(loss_arg) = args.first() {
        crate::type_errors::check_loss_consumer(name, loss_arg, env)?;
    }
    let result = if name == "momentum_sgd" {
        eval_momentum_sgd(args, env)?
    } else {
        eval_adam(args, env)?
    };
    crate::fncall_trace::push_array_event(trace, name, span, vec![], &result);
    Ok(Value::Array(result))
}

/// `momentum_sgd(loss_expr, params, lr, beta)` built-in.
///
/// Classical momentum SGD: per-param velocity buffer `v` is
/// updated as `v_{t+1} = beta * v_t + grad`, then the weight
/// is updated as `w_{t+1} = w_t - lr * v_{t+1}`.
pub(crate) fn eval_momentum_sgd(
    args: &[Expr],
    env: &mut Environment,
) -> Result<DenseArray, EvalError> {
    if args.len() != 4 {
        return Err(EvalError::BadArity {
            func: "momentum_sgd".into(),
            expected: 4,
            got: args.len(),
        });
    }
    let loss_expr = args[0].clone();
    let param_names = crate::grad::collect_params(&args[1], env, "momentum_sgd")?;
    let scalar_arg = |expr: &Expr, env: &mut Environment| -> Result<f64, EvalError> {
        let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
        if arr.rank() != 0 {
            return Err(EvalError::Unsupported(
                "momentum_sgd: lr and beta must be scalars".into(),
            ));
        }
        Ok(arr.data()[0])
    };
    let lr = scalar_arg(&args[2], env)?;
    let beta = scalar_arg(&args[3], env)?;

    // Saga E4 step 006: resident path (see eval_adam's twin hook).
    if let Some(r) =
        crate::grad_optim_resident::try_momentum(&loss_expr, &param_names, lr, beta, env)
    {
        return r;
    }

    let mut grads = crate::grad::eval_grads_batch(&loss_expr, env)?;
    for name in &param_names {
        if !env.is_param(name) {
            return Err(EvalError::Unsupported(format!(
                "momentum_sgd: '{name}' is not a tracked parameter"
            )));
        }
        // Saga 15 step 001: frozen params keep their gradients
        // flowing through the tape but the optimizer update is
        // suppressed -- the weight stays put.
        if env.is_frozen(name) {
            continue;
        }
        let g = grads.remove(name).ok_or_else(|| {
            EvalError::Unsupported(format!("momentum_sgd: '{name}' is not a tracked parameter"))
        })?;
        let key = ("momentum_sgd".to_string(), name.clone(), "v".to_string());
        let v_old = env
            .optim_state
            .buffers
            .get(&key)
            .cloned()
            .unwrap_or_else(|| DenseArray::zeros(g.shape().clone()));
        if v_old.shape() != g.shape() {
            return Err(EvalError::Unsupported(format!(
                "momentum_sgd: stored velocity shape mismatch for '{name}'"
            )));
        }
        let v_data: Vec<f64> = v_old
            .data()
            .iter()
            .zip(g.data().iter())
            .map(|(vo, gv)| beta * vo + gv)
            .collect();
        let v_new =
            DenseArray::new(v_old.shape().clone(), v_data).expect("velocity shape matches grad");
        env.optim_state.buffers.insert(key, v_new.clone());

        let w = env.get(name).cloned().expect("param exists in environment");
        let w_data: Vec<f64> = w
            .data()
            .iter()
            .zip(v_new.data().iter())
            .map(|(wv, vv)| wv - lr * vv)
            .collect();
        let w_new =
            DenseArray::new(w.shape().clone(), w_data).expect("weight shape matches velocity");
        env.set(name.clone(), w_new);
    }
    *env.optim_state
        .steps
        .entry("momentum_sgd".into())
        .or_insert(0) += 1;
    Ok(DenseArray::from_scalar(0.0))
}

/// `adam(loss_expr, params, lr, b1, b2, eps)` built-in.
///
/// Standard Adam with bias correction. Per-param first/second moment
/// buffers `m`, `v` and a per-optimizer step counter `t` live in
/// `OptimizerState`. At each call (with `t` post-incremented to start
/// at 1):
///
/// ```text
///     g     = grad(loss_expr, w)
///     m     = b1*m + (1 - b1)*g
///     v     = b2*v + (1 - b2)*g*g
///     mhat  = m / (1 - b1^t)
///     vhat  = v / (1 - b2^t)
///     w     = w - lr * mhat / (sqrt(vhat) + eps)
/// ```
pub(crate) fn eval_adam(args: &[Expr], env: &mut Environment) -> Result<DenseArray, EvalError> {
    if args.len() != 6 {
        return Err(EvalError::BadArity {
            func: "adam".into(),
            expected: 6,
            got: args.len(),
        });
    }
    let loss_expr = args[0].clone();
    let param_names = crate::grad::collect_params(&args[1], env, "adam")?;
    let scalar_arg = |expr: &Expr, env: &mut Environment| -> Result<f64, EvalError> {
        let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
        if arr.rank() != 0 {
            return Err(EvalError::Unsupported(
                "adam: lr/b1/b2/eps must be scalars".into(),
            ));
        }
        Ok(arr.data()[0])
    };
    let lr = scalar_arg(&args[2], env)?;
    let b1 = scalar_arg(&args[3], env)?;
    let b2 = scalar_arg(&args[4], env)?;
    let eps = scalar_arg(&args[5], env)?;

    // Step counter is 1-based: bump first, then read.
    let t = {
        let entry = env.optim_state.steps.entry("adam".into()).or_insert(0);
        *entry += 1;
        *entry
    };
    let bc1 = 1.0 - b1.powi(t as i32);
    let bc2 = 1.0 - b2.powi(t as i32);

    // GPU fast path: under a GPU device, try this build's registered
    // step (head-only LoRA, then the board-policy MLP). Either `None`
    // (unrecognized model) falls through to the CPU tape below. The whole
    // block is conditional on a GPU backend being configured -- on a
    // CPU-only build the `gpu_step` module/field do not exist -- and it
    // names no backend: the specifics live behind `GpuAdamStep`.
    #[cfg(any(
        all(target_os = "macos", target_arch = "aarch64", feature = "mlx"),
        all(target_os = "linux", target_arch = "x86_64", feature = "cuda")
    ))]
    if env.device() != "cpu"
        && let Some(step) = env.gpu_step()
    {
        let hp = mlpl_eval_state::AdamHp {
            lr,
            b1,
            b2,
            eps,
            t: t as i32,
        };
        // Recognition is interpreter-coupled (demo_layout/mlp_layout read
        // models; extract_xy evals the X/Y exprs), so it stays here; the
        // GPU step then gets only the resolved layout + tensors.
        if let Some(layout) = crate::grad_optim_mlx_demo::demo_layout(&args[1], env)
            && let Some((x, y)) = crate::grad_optim_mlx_demo::extract_xy(&loss_expr, env)
        {
            return step.run_lora_step(&layout, &x, &y, &hp, env);
        }
        if let Some((l1, head)) = crate::grad_optim_mlx_mlp::mlp_layout(&args[1], env)
            && let Some((x, y)) = crate::grad_optim_mlx_demo::extract_xy(&loss_expr, env)
        {
            return step.run_mlp_step(&l1, &head, &x, &y, &hp, env);
        }
    }

    // Saga E4 step 006: the general resident path -- ONE tape per
    // step, weights + moments resident across the loop. Applies to
    // any model under device("mlx"); returns None on other devices
    // or when no backend registered.
    if let Some(r) = crate::grad_optim_resident::try_adam(
        &loss_expr,
        &param_names,
        &crate::grad_optim_resident::ResidentHp {
            lr,
            b1,
            b2,
            eps,
            bc1,
            bc2,
        },
        env,
    ) {
        return r;
    }

    // Reaching the CPU tape under a GPU device means no GPU fast path
    // matched this model (only head-only LoRA + the board-policy MLP are
    // GPU-accelerated today). Surface that to the user once per eval --
    // the fallback is otherwise silent (looks like the GPU is idle).
    if env.device() != "cpu" {
        let dev = env.device().to_string();
        env.push_notice_once(format!(
            "note: adam under device(\"{dev}\") ran on the CPU -- this model has no GPU \
             fast path yet (only head-only LoRA and the board-policy MLP are GPU-accelerated). \
             See docs/future-saga-gpu-training.md."
        ));
    }
    let mut grads = crate::grad::eval_grads_batch(&loss_expr, env)?;
    for name in &param_names {
        if !env.is_param(name) {
            return Err(EvalError::Unsupported(format!(
                "adam: '{name}' is not a tracked parameter"
            )));
        }
        // Saga 15 step 001: frozen params keep their gradients
        // flowing through the tape but the optimizer update is
        // suppressed -- the weight stays put.
        if env.is_frozen(name) {
            continue;
        }
        let g = grads.remove(name).ok_or_else(|| {
            EvalError::Unsupported(format!("adam: '{name}' is not a tracked parameter"))
        })?;
        let m_key = ("adam".to_string(), name.clone(), "m".to_string());
        let v_key = ("adam".to_string(), name.clone(), "v".to_string());
        let m_old = env
            .optim_state
            .buffers
            .get(&m_key)
            .cloned()
            .unwrap_or_else(|| DenseArray::zeros(g.shape().clone()));
        let v_old = env
            .optim_state
            .buffers
            .get(&v_key)
            .cloned()
            .unwrap_or_else(|| DenseArray::zeros(g.shape().clone()));
        if m_old.shape() != g.shape() || v_old.shape() != g.shape() {
            return Err(EvalError::Unsupported(format!(
                "adam: stored moment shape mismatch for '{name}'"
            )));
        }
        let m_data: Vec<f64> = m_old
            .data()
            .iter()
            .zip(g.data().iter())
            .map(|(mo, gv)| b1 * mo + (1.0 - b1) * gv)
            .collect();
        let v_data: Vec<f64> = v_old
            .data()
            .iter()
            .zip(g.data().iter())
            .map(|(vo, gv)| b2 * vo + (1.0 - b2) * gv * gv)
            .collect();
        let m_new = DenseArray::new(g.shape().clone(), m_data).expect("m shape matches grad");
        let v_new = DenseArray::new(g.shape().clone(), v_data).expect("v shape matches grad");

        let w = env.get(name).cloned().expect("param exists in environment");
        let w_data: Vec<f64> = w
            .data()
            .iter()
            .zip(m_new.data().iter())
            .zip(v_new.data().iter())
            .map(|((wv, mv), vv)| {
                let mhat = mv / bc1;
                let vhat = vv / bc2;
                wv - lr * mhat / (vhat.sqrt() + eps)
            })
            .collect();
        let w_new = DenseArray::new(w.shape().clone(), w_data).expect("weight shape matches grad");

        env.optim_state.buffers.insert(m_key, m_new);
        env.optim_state.buffers.insert(v_key, v_new);
        env.set(name.clone(), w_new);
    }
    Ok(DenseArray::from_scalar(0.0))
}
