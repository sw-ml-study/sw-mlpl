//! Engram parameter creation (split from `model_eval_engram.rs`,
//! saga E3 step 2): allocates the five NEAR-IDENTITY parameters --
//! zero memory table, small projections, negative gate bias -- so
//! an untrained engram is (numerically) a no-op on the residual
//! stream until training moves its rows.

use crate::env_api::{EnvDevice, EnvParams, EnvTags, EnvTensorDevice};
use mlpl_core::ValueTag;
use mlpl_engram_core::EngramSpec;

use crate::model_engram_values::engram_param_values;
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_env::Environment;
use mlpl_eval_types::EvalError;

/// Parameter dimensions bundle for the initializers.
pub(crate) struct EngramDims {
    pub rows: usize,
    pub head_dim: usize,
    pub retrieved: usize,
    pub hidden: usize,
}

/// Allocate id + parameters for a validated spec and return the
/// finished `ModelSpec::Engram`.
pub(crate) fn make_engram(env: &mut Environment, spec: EngramSpec) -> Result<ModelSpec, EvalError> {
    let dims = EngramDims {
        rows: spec
            .table_rows()
            .map_err(|e| EvalError::Unsupported(e.to_string()))?,
        head_dim: spec.head_dim,
        retrieved: spec
            .retrieved_width()
            .map_err(|e| EvalError::Unsupported(e.to_string()))?,
        hidden: spec.hidden_size,
    };
    let id = env.next_model_id;
    env.next_model_id += 1;
    let names = engram_param_names(id);
    init_engram_params(env, &names, &dims, spec.seed as f64, id)?;
    Ok(spec_to_model(names, spec))
}

/// Fold the five parameter names and the validated spec into the
/// `ModelSpec::Engram` variant.
fn spec_to_model(names: [String; 5], spec: EngramSpec) -> ModelSpec {
    let [memory, w_value, b_value, w_gate, b_gate] = names;
    ModelSpec::Engram {
        memory,
        w_value,
        b_value,
        w_gate,
        b_gate,
        hidden: spec.hidden_size,
        ngram_orders: spec.ngram_orders,
        heads: spec.heads_per_ngram,
        slots: spec.slots_per_head,
        head_dim: spec.head_dim,
        seed: spec.seed,
    }
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
