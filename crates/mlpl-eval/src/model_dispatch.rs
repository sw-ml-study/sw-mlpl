//! Built-in dispatch for the Saga 11 model DSL.
//!
//! `linear(in_dim, out_dim, seed)` creates a fresh `ModelSpec::Linear`
//! whose `W` and `b` parameters are stored in the environment under
//! generated names. `apply(model_ident, X)` looks up the model and
//! evaluates it on the given input.

use mlpl_array::{DenseArray, Shape};
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use crate::error::EvalError;
use crate::model::ModelSpec;

/// `linear(in_dim, out_dim, seed)`.
pub(crate) fn eval_linear(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if args.len() != 3 {
        return Err(EvalError::BadArity {
            func: "linear".into(),
            expected: 3,
            got: args.len(),
        });
    }
    let in_dim = scalar_usize(&args[0], env, "linear")?;
    let out_dim = scalar_usize(&args[1], env, "linear")?;
    let seed = scalar_f64(&args[2], env, "linear")?;

    let id = env.next_model_id;
    env.next_model_id += 1;
    let w_name = format!("__linear_W_{id}");
    let b_name = format!("__linear_b_{id}");

    // W <- randn(seed, [in_dim, out_dim]) * 0.5 (Xavier-ish small).
    let w_init = mlpl_runtime::call_builtin(
        "randn",
        vec![
            DenseArray::from_scalar(seed),
            DenseArray::new(Shape::new(vec![2]), vec![in_dim as f64, out_dim as f64])?,
        ],
    )?;
    let w_data: Vec<f64> = w_init.data().iter().map(|v| v * 0.5).collect();
    let w = DenseArray::new(Shape::new(vec![in_dim, out_dim]), w_data)?;
    env.set_param(w_name.clone(), w);

    let b = DenseArray::zeros(Shape::new(vec![1, out_dim]));
    env.set_param(b_name.clone(), b);

    Ok(ModelSpec::Linear {
        w: w_name,
        b: b_name,
    })
}

/// `apply(model_ident, X)`.
pub(crate) fn eval_apply(
    args: &[Expr],
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<DenseArray, EvalError> {
    if args.len() != 2 {
        return Err(EvalError::BadArity {
            func: "apply".into(),
            expected: 2,
            got: args.len(),
        });
    }
    let model_name = match &args[0] {
        Expr::Ident(n, _) => n.clone(),
        _ => {
            return Err(EvalError::Unsupported(
                "apply: first argument must be a model identifier".into(),
            ));
        }
    };
    let model = env
        .get_model(&model_name)
        .cloned()
        .ok_or_else(|| EvalError::UndefinedVariable(model_name.clone()))?;
    let x = crate::eval::eval_expr(&args[1], env, trace)?.into_array()?;
    apply_model(&model, &x, env)
}

fn apply_model(
    model: &ModelSpec,
    x: &DenseArray,
    env: &Environment,
) -> Result<DenseArray, EvalError> {
    match model {
        ModelSpec::Linear { w, b } => {
            let w_arr = env
                .get(w)
                .ok_or_else(|| EvalError::UndefinedVariable(w.clone()))?;
            let b_arr = env
                .get(b)
                .ok_or_else(|| EvalError::UndefinedVariable(b.clone()))?;
            let xw = x.matmul(w_arr)?;
            // Broadcast b ([1, out]) up to [n, out] via ones([n, 1]) @ b.
            let n = xw.shape().dims()[0];
            let ones = DenseArray::new(Shape::new(vec![n, 1]), vec![1.0; n])?;
            let b_broadcast = ones.matmul(b_arr)?;
            Ok(xw.apply_binop(&b_broadcast, |a, c| a + c)?)
        }
    }
}

fn scalar_f64(expr: &Expr, env: &mut Environment, func: &str) -> Result<f64, EvalError> {
    let arr = crate::eval::eval_expr(expr, env, &mut None)?.into_array()?;
    if arr.rank() != 0 {
        return Err(EvalError::Unsupported(format!(
            "{func}: expected a scalar argument"
        )));
    }
    Ok(arr.data()[0])
}

fn scalar_usize(expr: &Expr, env: &mut Environment, func: &str) -> Result<usize, EvalError> {
    let v = scalar_f64(expr, env, func)?;
    if v < 0.0 || v.fract() != 0.0 {
        return Err(EvalError::Unsupported(format!(
            "{func}: dimension must be a non-negative integer"
        )));
    }
    Ok(v as usize)
}
