//! Connect-mode URL log helper + the `window.__mlpl_ask` JS
//! bridge. Both are window-level side-effects, kept out of
//! `app_hooks` so that module stays under its function-count cap.

use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use yew::prelude::*;

/// Same-origin auto-connect: when the page has NO `?connect=` param
/// at all (absent, not the `off` sentinel), probe the page's own
/// origin for `/v1/devices`; if it answers like an mlpl-serve, adopt
/// the origin as the connect target via `history.replaceState`
/// BEFORE the app mounts -- the common one-server case then needs no
/// param. Static hosts (github.io, trunk) fail the probe and stay
/// browser-local. WASM-only; native is a no-op.
pub async fn apply_same_origin_autoconnect() {
    #[cfg(target_arch = "wasm32")]
    {
        let Some(win) = web_sys::window() else { return };
        let search = win.location().search().unwrap_or_default();
        // Any explicit param -- a URL or the off sentinel -- wins.
        if mlpl_web_eval::eval_url::parse_connect_url(&search).is_some() {
            return;
        }
        let Ok(origin) = win.location().origin() else {
            return;
        };
        let devices = mlpl_web_eval::devices::fetch_devices_at(&origin).await;
        if devices.is_empty() {
            return;
        }
        let sep = if search.is_empty() { "?" } else { "&" };
        let path = win.location().pathname().unwrap_or_default();
        let url = format!("{path}{search}{sep}connect={origin}");
        if let Ok(history) = win.history() {
            let _ = history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&url));
            web_sys::console::log_1(&format!("[mlpl-web] auto-connect: {origin}").into());
        }
    }
}

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

/// One-call bundle for the App component's trailing chrome hooks
/// (scroll tracking, post-splash focus, :ask wiring, global
/// keydown). A `#[hook]` so the rule-of-hooks holds; exists so
/// `app()` stays inside the function-LOC budget without hiding
/// any hook behind a condition.
#[hook]
pub fn use_chrome_hooks(
    active: &mlpl_web_render_types::state::ActiveContext,
    ui: &mlpl_web_render_types::state::UiState,
    onboarding: &crate::app_state::OnboardingState,
    callbacks: &mlpl_web_render_types::app_callbacks::AppCallbacks,
) {
    crate::app_hooks::use_scroll_effect(active.history.clone());
    crate::app_hooks::use_focus_after_splash(onboarding.show_splash.clone());
    use_ask_hook(callbacks.on_submit.clone());
    crate::app_hooks::use_global_keydown(
        ui.dialog_open.clone(),
        ui.lesson_idx.clone(),
        onboarding.show_tour.clone(),
        onboarding.show_splash.clone(),
        ui.show_3d.clone(),
    );
}
