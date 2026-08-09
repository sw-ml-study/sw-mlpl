//! `:describe` / `help` rendering for registered native
//! extension functions. Extension signatures come from the
//! registry as a bounded TOML doc (`returns` + `documentation`);
//! this renders them through the same one-line shape the builtin
//! catalog uses, e.g.
//! `hello:answer() -> i64 : Return the canonical extension answer.`

/// Describe a registered extension function, or `None` if `name`
/// is not one.
pub(crate) fn describe(name: &str) -> Option<String> {
    let (_, sig) = mlpl_extension_registry::signatures()
        .into_iter()
        .find(|(k, _)| k == name)?;
    let returns = toml_field(&sig, "returns").unwrap_or_else(|| "value".to_string());
    let mut out = format!("{name}() -> {returns} -- native extension");
    if let Some(doc) = toml_field(&sig, "documentation") {
        out.push_str(&format!("\n  {doc}"));
    }
    Some(out)
}

/// Extract a `key = "value"` string field from the bounded
/// signature TOML (a tiny scan -- the doc is provider-controlled
/// and single-level).
fn toml_field(sig: &str, key: &str) -> Option<String> {
    for line in sig.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key)
            && let Some(eq) = rest.trim_start().strip_prefix('=')
        {
            return Some(eq.trim().trim_matches('"').to_string());
        }
    }
    None
}
