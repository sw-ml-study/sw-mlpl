//! Saga 29 step 024: glossary popup host. Listens for the
//! window-level `mlpl-glossary-open` CustomEvent dispatched
//! by `[[term]]` clicks anywhere in the playground, and
//! renders a modal overlay with the matched glossary entry
//! (or an "unknown term" warning if the entry doesn't
//! exist).
//!
//! Decoupled from the renderer via the CustomEvent so no
//! callback plumbing is needed -- any rendered `[[term]]`
//! link anywhere in the app pops the same overlay.

use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use yew::prelude::*;

use crate::glossary_view::find_by_term;
use mlpl_web_path_body::render_body;

#[function_component(GlossaryPopupHost)]
pub fn glossary_popup_host() -> Html {
    let term = use_state(|| None::<String>);
    let term_for_open = term.clone();
    let term_for_keydown = term.clone();
    // Yew rule-of-hooks: `use_effect_with` must be called at the
    // top of the function-component body, not in a helper fn.
    use_effect_with((), move |_| {
        install_listeners(term_for_open, term_for_keydown);
        || ()
    });
    let Some(open_term) = (*term).clone() else {
        return html! {};
    };
    let on_close = {
        let term = term.clone();
        Callback::from(move |_| term.set(None))
    };
    let body = match find_by_term(&open_term) {
        Some(entry) => render_body(&entry.body),
        None => html! {
            <p class="path-step-error">{ format!("(no glossary entry named {open_term:?})") }</p>
        },
    };
    html! {
        <div class="glossary-popup-overlay" onclick={on_close.clone()}>
            <div class="glossary-popup-card" onclick={Callback::from(|e: MouseEvent| e.stop_propagation())}>
                <div class="glossary-popup-header">
                    <h3 class="glossary-popup-title">{ &open_term }</h3>
                    <button class="glossary-popup-close" onclick={on_close} title="Close (Esc)">{"\u{2715}"}</button>
                </div>
                <div class="glossary-popup-body">{ body }</div>
            </div>
        </div>
    }
}

/// Wire the two window-level listeners. Called from inside
/// the use_effect_with closure (not as a hook itself, so it
/// doesn't trip the rule-of-hooks). Closures are `forget()`d
/// because GlossaryPopupHost is mounted once for the whole
/// session; the leak is bounded.
fn install_listeners(
    term_for_open: UseStateHandle<Option<String>>,
    term_for_keydown: UseStateHandle<Option<String>>,
) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let open_closure = make_open_closure(term_for_open);
    let _ = window.add_event_listener_with_callback(
        "mlpl-glossary-open",
        open_closure.as_ref().unchecked_ref(),
    );
    open_closure.forget();
    let keydown_closure = make_esc_closure(term_for_keydown);
    let _ = window
        .add_event_listener_with_callback("keydown", keydown_closure.as_ref().unchecked_ref());
    keydown_closure.forget();
    web_sys::console::log_1(&"GlossaryPopupHost listeners installed".into());
}

fn make_open_closure(term: UseStateHandle<Option<String>>) -> Closure<dyn FnMut(web_sys::Event)> {
    Closure::wrap(Box::new(move |e: web_sys::Event| {
        web_sys::console::log_1(&"mlpl-glossary-open listener fired".into());
        let Ok(custom) = e.dyn_into::<web_sys::CustomEvent>() else {
            web_sys::console::warn_1(&"event was not a CustomEvent".into());
            return;
        };
        let Some(detail) = custom.detail().as_string() else {
            web_sys::console::warn_1(&"event detail was not a string".into());
            return;
        };
        web_sys::console::log_1(&format!("opening glossary popup for: {detail}").into());
        term.set(Some(detail));
    }))
}

fn make_esc_closure(
    term: UseStateHandle<Option<String>>,
) -> Closure<dyn FnMut(web_sys::KeyboardEvent)> {
    Closure::wrap(Box::new(move |e: web_sys::KeyboardEvent| {
        if e.key() == "Escape" {
            term.set(None);
        }
    }))
}
