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
use crate::path_body::render_body;

#[function_component(GlossaryPopupHost)]
pub fn glossary_popup_host() -> Html {
    let term = use_state(|| None::<String>);
    install_listener(&term);
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

fn install_listener(term: &UseStateHandle<Option<String>>) {
    let term_for_open = term.clone();
    let term_for_keydown = term.clone();
    use_effect_with((), move |_| {
        // Install both listeners on `window`. We can't cleanly
        // remove them on teardown without keeping owned
        // function pointers, so we just leak the closures via
        // `forget()` -- there is only ever one
        // GlossaryPopupHost mounted, so the leak is bounded.
        if let Some(window) = web_sys::window() {
            let open_closure = make_open_closure(term_for_open);
            let _ = window.add_event_listener_with_callback(
                "mlpl-glossary-open",
                open_closure.as_ref().unchecked_ref(),
            );
            open_closure.forget();
            let keydown_closure = make_esc_closure(term_for_keydown);
            let _ = window.add_event_listener_with_callback(
                "keydown",
                keydown_closure.as_ref().unchecked_ref(),
            );
            keydown_closure.forget();
        }
        || ()
    });
}

fn make_open_closure(term: UseStateHandle<Option<String>>) -> Closure<dyn FnMut(web_sys::Event)> {
    Closure::wrap(Box::new(move |e: web_sys::Event| {
        let Ok(custom) = e.dyn_into::<web_sys::CustomEvent>() else {
            return;
        };
        let Some(detail) = custom.detail().as_string() else {
            return;
        };
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
