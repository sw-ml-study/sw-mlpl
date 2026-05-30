//! Connect-mode URL log helper + the `window.__mlpl_ask` JS
//! bridge. Both are window-level side-effects, kept out of
//! `app_hooks` so that module stays under its function-count cap.

use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use yew::prelude::*;

/// Log the connect-mode URL if one is present in the page
/// query string. WASM-only side-effect; on native this is a
/// no-op stub so call sites stay target-agnostic.
pub fn log_connect_mode() {
    #[cfg(target_arch = "wasm32")]
    if let Some(url) = mlpl_web_eval::eval::current_connect_url_from_window() {
        web_sys::console::log_1(&format!("[mlpl-web] connect mode: {url}").into());
        // Prime the server-configured Ollama default (host + model)
        // so `:ask` can fall back to it without a `?ollama=`/`?model=`
        // override. Fire-and-forget; failures leave the built-in
        // defaults in place.
        mlpl_web_eval::ollama_fetch::prime_ollama_default(url);
    }
}

/// Expose `window.__mlpl_ask(question)` so JS (the 3D inspector's
/// "Ask" affordance) can submit a `:ask <question>` line through
/// the normal REPL pipeline -- which, in connect mode, routes to
/// the server-side LLM with the selection + REPL context attached.
#[hook]
pub fn use_ask_hook(on_submit: Callback<String>) {
    use_effect_with((), move |_| {
        if let Some(window) = web_sys::window() {
            let closure = Closure::wrap(Box::new(move |q: wasm_bindgen::JsValue| {
                if let Some(text) = q.as_string() {
                    on_submit.emit(format!(":ask {text}"));
                }
            }) as Box<dyn FnMut(wasm_bindgen::JsValue)>);
            let _ = js_sys::Reflect::set(
                &window,
                &wasm_bindgen::JsValue::from_str("__mlpl_ask"),
                closure.as_ref().unchecked_ref(),
            );
            closure.forget();
        }
        || ()
    });
}
