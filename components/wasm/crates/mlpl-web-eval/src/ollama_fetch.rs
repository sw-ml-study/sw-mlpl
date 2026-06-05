//! Connect-mode fetches of the server's Ollama settings: the model
//! list (`GET /v1/ollama/tags`) backing `:connect list`, and the
//! default host+model (`GET /v1/ollama/config`) primed on connect so
//! `:ask` can fall back to the server config when the page carries
//! no `?ollama=` / `?model=` override. Phase 0 part 2 of the
//! local-gpu-agentic saga. WASM-only (the whole module is elided on
//! native, like `eval_wasm`).

#![cfg(target_arch = "wasm32")]

use crate::eval::ResultCb;

thread_local! {
    // (host, model) from GET /v1/ollama/config, primed once on connect.
    static OLLAMA_DEFAULT: std::cell::RefCell<Option<(String, String)>> =
        const { std::cell::RefCell::new(None) };
    // Session model override from `:connect set <model>`.
    static OLLAMA_SELECTED: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// The server-configured default `(host, model)` if it has been
/// primed, else `None`. `:ask` uses it when the page carries no
/// `?ollama=` / `?model=` override.
pub fn ollama_default() -> Option<(String, String)> {
    OLLAMA_DEFAULT.with(|c| c.borrow().clone())
}

/// Set the session model (`:connect set <model>`); `:ask` uses it over
/// any `?model=` page override and the server default.
pub fn set_selected_model(name: &str) {
    OLLAMA_SELECTED.with(|c| *c.borrow_mut() = Some(name.to_string()));
}

/// The session model override from `:connect set`, if any.
pub fn selected_model() -> Option<String> {
    OLLAMA_SELECTED.with(|c| c.borrow().clone())
}

/// Fire-and-forget `GET <base>/v1/ollama/config`, caching
/// `(host, model)` for later `:ask` defaulting. Called once when
/// connect mode is detected; failures leave the default unset (so
/// `:ask` falls back to its built-in constants).
pub fn prime_ollama_default(base_url: String) {
    wasm_bindgen_futures::spawn_local(async move {
        let url = format!("{}/v1/ollama/config", base_url.trim_end_matches('/'));
        let Ok(resp) = gloo::net::http::Request::get(&url).send().await else {
            return;
        };
        let Ok(body) = resp.json::<serde_json::Value>().await else {
            return;
        };
        let host = body.get("host").and_then(|v| v.as_str());
        let model = body.get("model").and_then(|v| v.as_str());
        if let (Some(h), Some(m)) = (host, model) {
            OLLAMA_DEFAULT.with(|c| *c.borrow_mut() = Some((h.to_string(), m.to_string())));
        }
    });
}

/// Async `GET <base>/v1/ollama/tags`; fires `on_result` with a
/// human-readable model listing (or an `error:`-prefixed message)
/// for the `:connect list [host]` REPL command. `host` (the
/// optional per-call override) is forwarded as `?host=` and must
/// be allow-listed on the server.
pub fn fetch_ollama_models(base_url: String, host: Option<String>, on_result: ResultCb) {
    wasm_bindgen_futures::spawn_local(async move {
        on_result(ollama_models_text(&base_url, host.as_deref()).await);
    });
}

async fn ollama_models_text(base_url: &str, host: Option<&str>) -> String {
    let suffix = host.map(|h| format!("?host={h}")).unwrap_or_default();
    let url = format!("{}/v1/ollama/tags{suffix}", base_url.trim_end_matches('/'));
    let resp = match gloo::net::http::Request::get(&url).send().await {
        Ok(r) => r,
        Err(e) => return format!("error: {e}"),
    };
    if !resp.ok() {
        return format!("error: /v1/ollama/tags returned {}", resp.status());
    }
    let body: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(e) => return format!("error: decode: {e}"),
    };
    format_model_list(&body)
}

/// Render the model list: each model by size (desc), marking the one
/// `:ask` would use (the `:connect set` selection, else the server
/// default). The footer points at `:connect set <name>`.
fn format_model_list(body: &serde_json::Value) -> String {
    let current = selected_model()
        .or_else(|| ollama_default().map(|(_, m)| m))
        .unwrap_or_default();
    let mut rows: Vec<(String, u64)> = body["models"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| Some((m["name"].as_str()?.to_string(), m["size"].as_u64()?)))
                .collect()
        })
        .unwrap_or_default();
    if rows.is_empty() {
        return "(no Ollama models on the configured host)".to_string();
    }
    rows.sort_by_key(|(_, s)| std::cmp::Reverse(*s));
    let lines: Vec<String> = rows
        .iter()
        .map(|(n, s)| {
            let mark = if *n == current { "   <- current" } else { "" };
            format!("  {n:<30} {:>5.1} GB{mark}", *s as f64 / 1e9)
        })
        .collect();
    format!(
        "Ollama models ({}):\n{}\n(select with `:connect set <name>`)",
        rows.len(),
        lines.join("\n")
    )
}
