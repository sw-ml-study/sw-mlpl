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
    web_sys::console::log_1(&format!("glossary link clicked: {term}").into());
    let Some(window) = web_sys::window() else {
        web_sys::console::warn_1(&"no window for glossary dispatch".into());
        return;
    };
    let init = web_sys::CustomEventInit::new();
    init.set_detail(&wasm_bindgen::JsValue::from_str(term));
    init.set_bubbles(true);
    let Ok(event) = web_sys::CustomEvent::new_with_event_init_dict("mlpl-glossary-open", &init)
    else {
        web_sys::console::warn_1(&"failed to construct CustomEvent".into());
        return;
    };
    match window
        .unchecked_into::<web_sys::EventTarget>()
        .dispatch_event(&event)
    {
        Ok(true) => {}
        Ok(false) => web_sys::console::warn_1(&"event default-prevented by listener".into()),
        Err(_) => web_sys::console::warn_1(&"dispatch_event returned err".into()),
    }
}
