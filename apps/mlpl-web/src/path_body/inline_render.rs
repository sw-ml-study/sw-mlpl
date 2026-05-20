//! Rendering side of the inline-span parser. Keeps the
//! parser file (inline.rs) under the per-module function-
//! count budget after step 024 added glossary-link
//! dispatch + a test battery.

use wasm_bindgen::JsCast;
use yew::prelude::*;

use super::inline::Span;

pub(super) fn render_span(span: Span) -> Html {
    match span {
        Span::Text(s) => html! { { s } },
        Span::Code(s) => html! { <code class="path-body-code">{ s }</code> },
        Span::Bold(s) => html! { <strong>{ s }</strong> },
        Span::Emph(s) => html! { <em>{ s }</em> },
        Span::Glossary(term) => render_glossary_link(term),
    }
}

fn render_glossary_link(term: String) -> Html {
    let term_for_click = term.clone();
    let onclick = Callback::from(move |e: MouseEvent| {
        e.prevent_default();
        dispatch_glossary_open(&term_for_click);
    });
    html! {
        <button class="glossary-link" type="button" {onclick} title={format!("Open glossary entry: {term}")}>
            { term }
        </button>
    }
}

fn dispatch_glossary_open(term: &str) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let init = web_sys::CustomEventInit::new();
    init.set_detail(&wasm_bindgen::JsValue::from_str(term));
    init.set_bubbles(true);
    let Ok(event) = web_sys::CustomEvent::new_with_event_init_dict("mlpl-glossary-open", &init)
    else {
        return;
    };
    let _ = window
        .unchecked_into::<web_sys::EventTarget>()
        .dispatch_event(&event);
}
