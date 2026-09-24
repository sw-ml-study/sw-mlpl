//! Parameter ROLES of the built-in layers: the stable, documented way
//! to address a layer's weights (`get_param(model, "Wq")`) instead of
//! the generated parameter names, which depend on construction order.
//! Each layer's role names pair, in order, with `ModelSpec::params()`,
//! so the names are defined once per layer here.

use mlpl_eval_core::model::ModelSpec;

/// The role names of a single layer, in `params()` order. Composite and
/// parameter-free layers own none.
fn role_names(spec: &ModelSpec) -> &'static [&'static str] {
    match spec {
        ModelSpec::Linear { b: Some(_), .. } => &["W", "b"],
        ModelSpec::Linear { b: None, .. } => &["W"],
        ModelSpec::Attention { .. } => &["Wq", "Wk", "Wv", "Wo"],
        ModelSpec::Embedding { .. } => &["table"],
        ModelSpec::LinearLora { .. } => &["W", "b", "A", "B"],
        ModelSpec::Engram { .. } => &["memory", "W_value", "b_value", "W_gate", "b_gate"],
        _ => &[],
    }
}

/// Every `(role, parameter name)` of `spec`, layer by layer in
/// application (depth-first) order.
fn owned_roles(spec: &ModelSpec) -> Vec<(&'static str, String)> {
    match spec {
        ModelSpec::Chain(children) => children.iter().flat_map(owned_roles).collect(),
        ModelSpec::Residual(inner) => owned_roles(inner),
        leaf => role_names(leaf)
            .iter()
            .copied()
            .zip(leaf.params())
            .collect(),
    }
}

/// The parameter name of the `k`-th (0-based, application order) layer
/// that owns `role`. `Err` is a message naming the roles the model does
/// own.
pub fn param_for_role(spec: &ModelSpec, role: &str, k: usize) -> Result<String, String> {
    let owned = owned_roles(spec);
    if let Some((_, name)) = owned.iter().filter(|(r, _)| *r == role).nth(k) {
        return Ok(name.clone());
    }
    let mut roles: Vec<&'static str> = owned.iter().map(|(r, _)| *r).collect();
    roles.dedup();
    Err(format!(
        "no layer #{k} with role \"{role}\"; this model's roles: {}",
        roles.join(", ")
    ))
}
