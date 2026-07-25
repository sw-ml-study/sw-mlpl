//! Page-context readers for `:ask`: URL query parameters, the
//! selected 3D sculpture, and the Ollama endpoint/model resolution.
//! Browser-only, like `prompt.rs`.

#![cfg(target_arch = "wasm32")]

/// Default Ollama endpoint + model backing the `:ask` shortcut.
/// Overridable per page via `?ollama=<url>` / `?model=<name>`;
/// mlpl-serve runs the `llm_call` server-side, so the target host
/// just has to be reachable from the server (no browser CORS).
const ASK_URL: &str = "http://localhost:11434";
const ASK_MODEL: &str = "qwen2.5:0.5b";

/// Read a URL query parameter, falling back to `default` when it
/// is absent or empty. (`name` is a fixed literal, never user
/// input, so the inlined script is safe.)
pub(crate) fn query_param(name: &str, default: &str) -> String {
    let script = format!("new URLSearchParams(location.search).get('{name}') || ''");
    js_sys::eval(&script)
        .ok()
        .and_then(|v| v.as_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// Plain-text summary of the sculpture the user is currently
/// inspecting (via the `window.__stage3d_context()` JS hook), or
/// empty when nothing is selected.
pub(crate) fn selection_context() -> String {
    js_sys::eval("window.__stage3d_context ? window.__stage3d_context() : ''")
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default()
}

/// Resolve the `(host, model)` for `:ask`. Host: `?ollama=` page
/// override, else the connect-primed default, else the constant. Model:
/// a `:connect set <model>` session pick wins, then `?model=`, then the
/// connect-primed default, then the constant.
pub(crate) fn ask_endpoint() -> (String, String) {
    let (def_url, def_model) = mlpl_web_eval::ollama_fetch::ollama_default()
        .unwrap_or_else(|| (ASK_URL.to_string(), ASK_MODEL.to_string()));
    let model = mlpl_web_eval::ollama_fetch::selected_model()
        .unwrap_or_else(|| query_param("model", &def_model));
    (query_param("ollama", &def_url), model)
}
