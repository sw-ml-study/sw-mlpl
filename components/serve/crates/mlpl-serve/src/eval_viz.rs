//! Phase 1c (local-gpu-agentic): extract viz data (shape, flat
//! values, string list) from an evaluated `Value` so the eval
//! response can carry it back. This lets the connect-mode web UI
//! emit 3D sculptures for server-evaluated results -- "the 3D view
//! shows everything regardless of where the work ran" -- without
//! the client re-evaluating. `viz_node` (attention/Sankey) is a
//! later part; this part covers the common tensor/grid/bar case.

use mlpl_eval::Value;

use crate::handlers::EvalResponse;
use crate::viz_storage::AttachedViz;

/// Cap on the number of flat values echoed in the response. The 3D
/// stage only renders small tensors as grids/bars; a large tensor
/// (a trained weight matrix, a long activation) would bloat the
/// JSON for no visual benefit, so we omit its values past the cap
/// (the shape still rides back).
const VIZ_VALUE_CAP: usize = 4096;

/// `(shape, values, string_list)` for `value`. Arrays yield their
/// dims + (capped) flat f64 data; string lists yield their items;
/// everything else yields empties (the client falls back to the
/// display string only).
pub(crate) fn value_viz(value: &Value) -> (Vec<usize>, Option<Vec<f64>>, Option<Vec<String>>) {
    match value {
        Value::Array(da) => {
            let data = da.data();
            let values = (data.len() <= VIZ_VALUE_CAP).then(|| data.to_vec());
            (da.shape().dims().to_vec(), values, None)
        }
        Value::StrList { items } => (vec![items.len()], None, Some(items.clone())),
        _ => (Vec::new(), None, None),
    }
}

/// Assemble the eval response: the formatted value + kind + any
/// attached SVG, plus the Phase 1c viz data (`value_viz`). Lives
/// here (not in the at-capacity handlers module) so `eval_handler`
/// stays under the function-LOC budget.
pub(crate) fn build_eval_response(
    value: &Value,
    kind: &'static str,
    formatted: String,
    attached: AttachedViz,
) -> EvalResponse {
    let (shape, values, string_list) = value_viz(value);
    let viz_node = match value {
        Value::Model(spec) => Some(mlpl_model_viz::model_to_viz_node(spec)),
        _ => None,
    };
    EvalResponse {
        value: formatted,
        kind,
        viz_url: attached.url,
        viz_local_path: attached.local_path,
        shape,
        values,
        string_list,
        viz_node,
    }
}
