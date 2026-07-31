//! `engram(hidden, ngrams, heads, slots, head_dim, seed)` -- the
//! Engram layer constructor (saga E2 step 1; design in
//! docs/engram-sagas-plan.md). Validates through mlpl-engram-core's
//! `EngramSpec` (the same accounting `:describe` prints); parameter
//! creation lives in `model_engram_init` / `model_engram_values`
//! (saga E3 step 2 split).

use mlpl_engram_core::EngramSpec;
use mlpl_parser::Expr;

use crate::model_dispatch_scalar::scalar_usize;
use crate::model_engram_init::make_engram;
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
    make_engram(env, spec)
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
