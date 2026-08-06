//! Phase 1c (local-gpu-agentic): extract viz data (shape, flat
//! values, string list) from an evaluated `Value` so the eval
//! response can carry it back. This lets the connect-mode web UI
//! emit 3D sculptures for server-evaluated results -- "the 3D view
//! shows everything regardless of where the work ran" -- without
//! the client re-evaluating. `viz_node` (attention/Sankey) is a
//! later part; this part covers the common tensor/grid/bar case.

use mlpl_eval::Value;

use axum::Json;
use axum::response::sse::Event;
use serde::{Deserialize, Serialize};

use crate::store::AttachedViz;

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
pub fn build_eval_response(
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

// ---- Wire/request/response types shared across the crate family ----

#[derive(Deserialize)]
pub struct EvalRequest {
    pub program: String,
}

#[derive(Serialize)]
pub struct EvalResponse {
    pub value: String,
    pub kind: &'static str,
    /// Saga 21.5 step 004: when the eval returned an SVG-shaped
    /// string, the server stashes the bytes in the `viz_storage`
    /// content-addressed store and surfaces the URL here.
    /// `None` (skipped on serialization) for non-SVG results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viz_url: Option<String>,
    /// Saga 21.5 step 004: server-side path inside
    /// `MLPL_CACHE_DIR` (when set) where the same SVG was also
    /// written. The dev-loopback case (`mlpl-serve` and
    /// `mlpl-repl` on one host) lets the client open the file
    /// directly; absent when no cache dir is configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viz_local_path: Option<String>,
    /// Phase 1c: viz data so the connect-mode web UI can emit a 3D
    /// sculpture for a server-evaluated result. `shape` empty +
    /// `values`/`string_list` absent for non-array/list results.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub shape: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub string_list: Option<Vec<String>>,
    /// Phase 1c part 2: a model's Sankey decomposition, so a model
    /// evaluated in connect mode renders its diagram. `None` for
    /// non-model results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viz_node: Option<mlpl_web_viz_ir::VizNode>,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// One frame on the SSE stream. The first frame is always
/// `Ready`. Zero-or-more `Metric` frames may follow (one per
/// `_metric`-suffixed scalar binding per `train { }` iteration).
/// The stream ends with exactly one terminal frame, either
/// `Done` (success) or `Error` (any `EvalError`).
#[derive(Debug, Serialize)]
#[serde(tag = "event", content = "data", rename_all = "lowercase")]
pub enum SseEvent {
    Ready,
    Metric {
        name: String,
        step: usize,
        value: f64,
    },
    /// One live tensor frame from the `emit_frame(name, step, x)`
    /// builtin (Game of Life saga step 4) -- the whole-tensor
    /// analog of `Metric`. Carries shape + flat values so the
    /// client can rebuild and render the board mid-eval.
    Frame {
        name: String,
        step: usize,
        shape: Vec<usize>,
        values: Vec<f64>,
    },
    Done {
        value: String,
        kind: &'static str,
        /// Saga 21.5 step 004: when the eval returned an
        /// SVG-shaped string, this carries the URL the
        /// `/v1/viz` storage minted for it. `None` (omitted
        /// from JSON) for non-SVG results.
        #[serde(skip_serializing_if = "Option::is_none")]
        viz_url: Option<String>,
        /// Saga 21.5 step 004: server-side `MLPL_CACHE_DIR`
        /// path for the same SVG, when configured.
        #[serde(skip_serializing_if = "Option::is_none")]
        viz_local_path: Option<String>,
    },
    /// Saga 21.5 step 003: terminal frame for cooperative
    /// cancellation. Mirrors `EvalError::Cancelled`'s `step`
    /// and `partial_losses` so clients can render the partial
    /// loss curve without falling back to a second
    /// `:vars`/inspect round-trip.
    Cancelled {
        step: usize,
        partial_losses: Vec<f64>,
    },
    Error {
        error: String,
    },
}

pub fn json_err(msg: impl Into<String>) -> Json<ErrorResponse> {
    Json(ErrorResponse { error: msg.into() })
}

pub fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Array(_) => "array",
        Value::Str(_) => "string",
        Value::Model(_) => "model",
        Value::Tokenizer(_) => "tokenizer",
        Value::DeviceTensor { .. } => "device-tensor",
        Value::BuiltinRef { .. } => "builtin-ref",
        Value::UserFnRef { .. } => "user-fn-ref",
        Value::Record { .. } => "record",
        Value::StrList { .. } => "string-list",
        Value::Result { .. } => "result",
    }
}

impl SseEvent {
    /// Convert one `SseEvent` to an axum `Event` carrying the
    /// `event:` line and a `data:` JSON payload. For variants
    /// without a body (`Ready`), `data:` is an empty JSON
    /// object so clients that always parse `data` do not need
    /// a special case.
    pub fn to_axum_event(&self) -> Event {
        let (name, data) = match self {
            Self::Ready => ("ready".to_string(), serde_json::json!({})),
            Self::Metric { name, step, value } => (
                "metric".to_string(),
                serde_json::json!({"name": name, "step": step, "value": value}),
            ),
            Self::Frame {
                name,
                step,
                shape,
                values,
            } => (
                "frame".to_string(),
                serde_json::json!({"name": name, "step": step, "shape": shape, "values": values}),
            ),
            Self::Done {
                value,
                kind,
                viz_url,
                viz_local_path,
            } => {
                let mut payload = serde_json::json!({"value": value, "kind": kind});
                if let Some(u) = viz_url {
                    payload["viz_url"] = serde_json::Value::String(u.clone());
                }
                if let Some(p) = viz_local_path {
                    payload["viz_local_path"] = serde_json::Value::String(p.clone());
                }
                ("done".to_string(), payload)
            }
            Self::Cancelled {
                step,
                partial_losses,
            } => (
                "cancelled".to_string(),
                serde_json::json!({"step": step, "partial_losses": partial_losses}),
            ),
            Self::Error { error } => ("error".to_string(), serde_json::json!({"error": error})),
        };
        Event::default().event(name).data(data.to_string())
    }
}
