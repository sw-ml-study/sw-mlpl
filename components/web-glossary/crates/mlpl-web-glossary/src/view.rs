//! Glossary tab content for the `?` documentation dialog.
//!
//! Renders a search box plus a scrollable list over the parsed glossary
//! (`mlpl-glossary-data`). Typing in the search box jumps the list to the
//! best fuzzy match (exact / prefix / substring / plural / alias) via the
//! `mlpl-glossary-search` crate. Educational-first: a student looking for
//! "MLP" types M, L, P and watches the list scroll to MLP without a mouse
//! round-trip; "k-quants" still finds "K-quant".

use wasm_bindgen::JsCast;
use web_sys::HtmlInputElement;
use yew::prelude::*;

use mlpl_glossary_data::{GlossaryEntry, doc};

// Re-export the data-layer lookup so `mlpl_web_glossary::view::find_by_term`
// stays the stable path for the Paths walker (mlpl-web-paths) + path_refs test.
pub use mlpl_glossary_data::find_by_term;

#[function_component(GlossaryView)]
pub fn glossary_view() -> Html {
    let g = doc();
    let query = use_state(String::new);
    let on_input = {
        let query = query.clone();
        Callback::from(move |e: InputEvent| {
            let target: HtmlInputElement = e.target_unchecked_into();
            query.set(target.value());
        })
    };
    let q_for_effect = (*query).clone();
    use_effect_with(q_for_effect, move |q: &String| {
        scroll_to_match(q, &g.entries);
        || ()
    });

    let matched =
        mlpl_glossary_search::best_match(&query, g.entries.iter().map(|e| e.term.as_str()));
    let entries_html = g.entries.iter().enumerate().map(|(i, e)| {
        let class = if Some(i) == matched {
            "glossary-entry matched"
        } else {
            "glossary-entry"
        };
        html! {
            <div class={class} id={e.slug.clone()}>
                <h3>{ &e.term }</h3>
                <div class="glossary-body">{ mlpl_web_path_body::render_body(&e.body) }</div>
            </div>
        }
    });
    let placeholder = "Type to jump (e.g. M, L, P -> MLP)";
    html! {
        <div class="glossary-view">
            <div class="glossary-search">
                <input type="text" placeholder={placeholder} oninput={on_input} value={(*query).clone()} aria-label="Search glossary terms" />
            </div>
            <div class="glossary-content-scroll" id="glossary-scroll">
                <p class="glossary-intro">{ &g.intro }</p>
                { for entries_html }
            </div>
        </div>
    }
}

fn scroll_to_match(query: &str, entries: &[GlossaryEntry]) {
    let Some(idx) =
        mlpl_glossary_search::best_match(query, entries.iter().map(|e| e.term.as_str()))
    else {
        return;
    };
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };
    let Some(el) = document.get_element_by_id(&entries[idx].slug) else {
        return;
    };
    if let Ok(html_el) = el.dyn_into::<web_sys::HtmlElement>() {
        // `true` aligns the element's top with the scroll container's
        // top. ScrollIntoViewOptions (smooth + start) needs more
        // web-sys features than the crate's current set; the boolean
        // form is on every browser MLPL targets and lands fine here.
        html_el.scroll_into_view_with_bool(true);
    }
}
