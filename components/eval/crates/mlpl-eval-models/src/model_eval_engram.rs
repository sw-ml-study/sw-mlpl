//! `engram(hidden, ngrams, heads, slots, head_dim, seed)` -- the
//! Engram layer constructor (saga E2 step 1; design in
//! docs/engram-sagas-plan.md). Validates through mlpl-engram-core's
//! `EngramSpec` (the same accounting `:describe` prints), then
//! creates the layer's parameters NEAR-IDENTITY: a zero memory
//! table and small projections with a negative gate bias, so an
//! untrained engram is (numerically) a no-op on the residual
//! stream until training moves its rows.

use crate::env_api::{EnvDevice, EnvParams, EnvTags, EnvTensorDevice};
use mlpl_array::{DenseArray, Shape};
use mlpl_core::ValueTag;
use mlpl_engram_core::EngramSpec;
use mlpl_parser::Expr;

use crate::model_dispatch_scalar::scalar_usize;
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

/// Evaluate the six-argument constructor into a `ModelSpec::Engram`,
/// creating and tagging its five parameters.
///
/// # Errors
/// Bad arity, non-scalar arguments, or an invalid `EngramSpec`.
pub fn eval_engram(args: &[Expr], env: &mut Environment) -> Result<ModelSpec, EvalError> {
    if args.len() != 6 {
        return Err(EvalError::BadArity {
            func: "engram".into(),
            expected: 6,
            got: args.len(),
        });
    }
    let spec = parse_engram_spec(args, env)?;
    let (rows, retrieved) = (
        spec.table_rows()
            .map_err(|e| EvalError::Unsupported(e.to_string()))?,
        spec.retrieved_width()
            .map_err(|e| EvalError::Unsupported(e.to_string()))?,
    );
    let EngramSpec {
        hidden_size: hidden,
        ngram_orders: orders,
        heads_per_ngram: heads,
        slots_per_head: slots,
        head_dim,
        seed,
    } = spec;
    let id = env.next_model_id;
    env.next_model_id += 1;
    let names = engram_param_names(id);
    let dims = EngramDims {
        rows,
        head_dim,
        retrieved,
        hidden,
    };
    init_engram_params(env, &names, &dims, seed as f64, id)?;
    let [memory, w_value, b_value, w_gate, b_gate] = names;
    Ok(ModelSpec::Engram {
        memory,
        w_value,
        b_value,
        w_gate,
        b_gate,
        hidden,
        ngram_orders: orders,
        heads,
        slots,
        head_dim,
        seed,
    })
}

/// Evaluate + validate the six constructor arguments into a spec.
fn parse_engram_spec(args: &[Expr], env: &mut Environment) -> Result<EngramSpec, EvalError> {
    let spec = EngramSpec {
        hidden_size: scalar_usize(&args[0], env, "engram")?,
        ngram_orders: order_list(&args[1], env)?,
        heads_per_ngram: scalar_usize(&args[2], env, "engram")?,
        slots_per_head: scalar_usize(&args[3], env, "engram")?,
        head_dim: scalar_usize(&args[4], env, "engram")?,
        seed: scalar_usize(&args[5], env, "engram")? as u64,
    };
    spec.validate()
        .map_err(|e| EvalError::Unsupported(e.to_string()))?;
    Ok(spec)
}

/// Evaluate the n-gram orders argument (an array literal like
/// `[2, 3]`) into a list of usize orders.
fn order_list(arg: &Expr, env: &mut Environment) -> Result<Vec<usize>, EvalError> {
    let arr = mlpl_eval_env::dispatch_hook::eval_or_err(arg, env, &mut None)?.into_array()?;
    arr.data()
        .iter()
        .map(|&v| {
            if v < 2.0 || v.fract() != 0.0 {
                Err(EvalError::Unsupported(format!(
                    "engram: ngram orders must be integers >= 2, got {v}"
                )))
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

/// Parameter dimensions bundle for [`init_engram_params`].
struct EngramDims {
    rows: usize,
    head_dim: usize,
    retrieved: usize,
    hidden: usize,
}

/// The five parameter names for engram layer `id`.
fn engram_param_names(id: u64) -> [String; 5] {
    [
        format!("__engram_mem_{id}"),
        format!("__engram_Wv_{id}"),
        format!("__engram_bv_{id}"),
        format!("__engram_Wg_{id}"),
        format!("__engram_bg_{id}"),
    ]
}

/// Create + tag the five parameters (gate starts nearly closed at
/// sigmoid(-2) ~ 0.12 on a zero-memory value path, so the module is
/// a numeric no-op until training).
fn init_engram_params(
    env: &mut Environment,
    names: &[String; 5],
    dims: &EngramDims,
    seed: f64,
    id: u64,
) -> Result<(), EvalError> {
    let layer = format!("engram_{id}");
    let values = engram_param_values(dims, seed)?;
    let device = env.device().to_string();
    for (i, arr, role) in values {
        env.set_param(names[i].clone(), arr);
        env.set_tensor_device(names[i].clone(), device.clone());
        env.set_tag(
            names[i].clone(),
            ValueTag::Weight {
                layer: layer.clone(),
                name: (*role).to_string(),
            },
        );
    }
    Ok(())
}

/// The five initial parameter arrays with their table roles: zero
/// memory, 0.01-scaled randn projections, zero value bias, -2 gate
/// bias.
fn engram_param_values(
    dims: &EngramDims,
    seed: f64,
) -> Result<[(usize, DenseArray, &'static str); 5], EvalError> {
    let &EngramDims {
        rows,
        head_dim,
        retrieved,
        hidden,
    } = dims;
    let scaled_randn = |seed: f64, r: usize, c: usize| -> Result<DenseArray, EvalError> {
        let init = mlpl_runtime::call_builtin(
            "randn",
            vec![
                DenseArray::from_scalar(seed),
                DenseArray::new(Shape::new(vec![2]), vec![r as f64, c as f64])?,
            ],
        )?;
        let data: Vec<f64> = init.data().iter().map(|v| v * 0.01).collect();
        Ok(DenseArray::new(Shape::new(vec![r, c]), data)?)
    };
    Ok([
        (
            0,
            DenseArray::zeros(Shape::new(vec![rows, head_dim])),
            "memory",
        ),
        (1, scaled_randn(seed + 1.0, retrieved, hidden)?, "W_v"),
        (2, DenseArray::zeros(Shape::new(vec![1, hidden])), "b_v"),
        (3, scaled_randn(seed + 2.0, 2 * hidden, hidden)?, "W_g"),
        (
            4,
            DenseArray::new(Shape::new(vec![1, hidden]), vec![-2.0; hidden])?,
            "b_g",
        ),
    ])
}
