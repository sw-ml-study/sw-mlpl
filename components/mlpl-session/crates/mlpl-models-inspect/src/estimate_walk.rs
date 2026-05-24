//! `estimate_train` tree-walk helpers: count hidden-dim
//! widest + parameterized-node depth, and the per-step FLOP
//! estimator. Both walk the `ModelSpec` recursively.

use mlpl_env_traits::HasVars;
use mlpl_eval_core::model::ModelSpec;

/// Statistics collected by `accumulate_hidden_depth`.
#[derive(Default)]
pub(crate) struct Stats {
    pub(crate) params: f64,
    pub(crate) trainable: f64,
    pub(crate) hidden: f64,
    pub(crate) depth: f64,
}

/// Depth-first walk that updates `hidden` (widest dim
/// observed) and `depth` (count of parameterized nodes).
pub(crate) fn accumulate_hidden_depth<E: HasVars>(spec: &ModelSpec, env: &E, acc: &mut Stats) {
    match spec {
        ModelSpec::Linear { w, .. } => {
            let (i, o) = linear_dims(env, w);
            acc.hidden = acc.hidden.max(i.max(o));
            acc.depth += 1.0;
        }
        ModelSpec::Embedding { d_model, .. } | ModelSpec::Attention { d_model, .. } => {
            acc.hidden = acc.hidden.max(*d_model as f64);
            acc.depth += 1.0;
        }
        ModelSpec::LinearLora {
            in_dim, out_dim, ..
        } => {
            acc.hidden = acc.hidden.max((*in_dim).max(*out_dim) as f64);
            acc.depth += 1.0;
        }
        ModelSpec::Chain(children) => children
            .iter()
            .for_each(|c| accumulate_hidden_depth(c, env, acc)),
        ModelSpec::Residual(inner) => accumulate_hidden_depth(inner, env, acc),
        ModelSpec::Activation(_) | ModelSpec::RmsNorm { .. } => {}
    }
}

pub(crate) fn walk_flops_per_step<E: HasVars>(
    spec: &ModelSpec,
    env: &E,
    batch: f64,
    seq: f64,
) -> f64 {
    match spec {
        ModelSpec::Linear { w, .. } => {
            let (in_dim, out_dim) = linear_dims(env, w);
            2.0 * in_dim * out_dim * batch
        }
        ModelSpec::Embedding { vocab, d_model, .. } => {
            2.0 * batch * (*vocab as f64) * (*d_model as f64)
        }
        ModelSpec::Attention { d_model, .. } => {
            let d = *d_model as f64;
            8.0 * d * d * batch * seq + 4.0 * seq * seq * d * batch
        }
        ModelSpec::LinearLora {
            in_dim,
            out_dim,
            rank,
            ..
        } => {
            let i = *in_dim as f64;
            let o = *out_dim as f64;
            let r = *rank as f64;
            2.0 * i * o * batch + 2.0 * i * r * batch + 2.0 * r * o * batch
        }
        ModelSpec::Chain(children) => children
            .iter()
            .map(|c| walk_flops_per_step(c, env, batch, seq))
            .sum(),
        ModelSpec::Residual(inner) => walk_flops_per_step(inner, env, batch, seq),
        ModelSpec::Activation(_) | ModelSpec::RmsNorm { .. } => 0.0,
    }
}

fn linear_dims<E: HasVars>(env: &E, w_name: &str) -> (f64, f64) {
    match env.get(w_name) {
        Some(arr) if arr.rank() == 2 => {
            let d = arr.shape().dims();
            (d[0] as f64, d[1] as f64)
        }
        _ => (0.0, 0.0),
    }
}
