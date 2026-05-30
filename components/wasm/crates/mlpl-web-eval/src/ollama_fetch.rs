//! Connect-mode fetches of the server's Ollama settings: the model
//! list (`GET /v1/ollama/tags`) backing `:models ollama`, and the
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
}

/// The server-configured default `(host, model)` if it has been
/// primed, else `None`. `:ask` uses it when the page carries no
/// `?ollama=` / `?model=` override.
pub fn ollama_default() -> Option<(String, String)> {
    OLLAMA_DEFAULT.with(|c| c.borrow().clone())
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
/// for the `:models ollama` REPL command.
pub fn fetch_ollama_models(base_url: String, on_result: ResultCb) {
    wasm_bindgen_futures::spawn_local(async move {
        on_result(ollama_models_text(&base_url).await);
    });
}

async fn ollama_models_text(base_url: &str) -> String {
    let url = format!("{}/v1/ollama/tags", base_url.trim_end_matches('/'));
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
    let names: Vec<&str> = body["models"]
        .as_array()
        .map(|a| a.iter().filter_map(|m| m["name"].as_str()).collect())
        .unwrap_or_default();
    if names.is_empty() {
        "(no Ollama models on the configured host)".to_string()
    } else {
        format!("Ollama models ({}):\n  {}", names.len(), names.join("\n  "))
    }
}
