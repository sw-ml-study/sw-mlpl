//! Phase 1c part 1b: emit a 3D sculpture from a connect-mode eval
//! response. The server now returns shape/values/string_list (see
//! mlpl-serve `eval_viz`), so a server-evaluated result populates
//! the 3D view just like a local one -- "3D everywhere". WASM-only.

#![cfg(target_arch = "wasm32")]

use mlpl_web_viz_ir::VizNode;
use serde_json::Value as Json;

/// Emit a stage3d event from an eval-response `body` when it carries
/// array `shape` (+ optional flat `values` / `string_list`) or a
/// model `viz_node` (Sankey), so a server-evaluated result renders
/// as a 3D sculpture. `program` supplies the label and the `name =`
/// target. No-op for a bare string / `:ask` reply (no shape, no viz).
pub(crate) fn emit_from_response(program: &str, body: &Json) {
    let shape: Vec<usize> = body["shape"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default();
    let viz_node: Option<VizNode> = body
        .get("viz_node")
        .and_then(|v| serde_json::from_value(v.clone()).ok());
    if shape.is_empty() && viz_node.is_none() {
        return;
    }
    let values = body["values"]
        .as_array()
        .map(|a| a.iter().filter_map(Json::as_f64).collect());
    let string_list = body["string_list"].as_array().map(|a| {
        a.iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect()
    });
    let name = program
        .split('=')
        .next()
        .unwrap_or(program)
        .trim()
        .to_string();
    let info =
        mlpl_web_viz3d::events::build_shape_info_full(name, shape, values, string_list, viz_node);
    mlpl_web_viz3d::events::emit(&mlpl_web_viz3d::events::Stage3dEvent {
        step_idx: 0,
        label: program.to_string(),
        output: info,
    });
}
