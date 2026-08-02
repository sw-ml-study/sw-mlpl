//! Glossary tab content for the `?` documentation dialog.
//!
//! Renders a search box plus a scrollable list over the parsed glossary
//! (`mlpl-glossary-data`). Typing FILTERS the list to every matching
//! entry, best match first (exact / prefix / substring / plural / alias
//! via the `mlpl-glossary-search` crate); clearing the box restores the
//! full alphabetical list. Educational-first: a student looking for
//! attention variants types "attention" and sees ALL of them at once --
//! no scrolling past the first hit; "k-quants" still finds "K-quant".

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
    let on_input = query_input_callback(&query);
    // Reset the list scroll whenever the filter changes so results
    // are visible from the top.
    let q_for_effect = (*query).clone();
    use_effect_with(q_for_effect, |_| {
        if let Some(el) = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("glossary-scroll"))
        {
            el.set_scroll_top(0);
        }
        || ()
    });

    let body = if query.trim().is_empty() {
        html! { <>
            <p class="glossary-intro">{ &g.intro }</p>
            { for g.entries.iter().map(|e| entry_html(e, false)) }
        </> }
    } else {
        filtered_html(&query, &g.entries)
    };
    let placeholder = "Type to filter (e.g. attention)";
    html! {
        <div class="glossary-view">
            <div class="glossary-search">
                <input type="text" placeholder={placeholder} oninput={on_input} value={(*query).clone()} aria-label="Search glossary terms" />
            </div>
            <div class="glossary-content-scroll" id="glossary-scroll">
                { body }
            </div>
        </div>
    }
}

/// The filtered view: every matching entry, best match first (and
/// highlighted), or a no-match notice.
fn filtered_html(query: &str, entries: &[GlossaryEntry]) -> Html {
    let hits = mlpl_glossary_search::all_matches(query, entries.iter().map(|e| e.term.as_str()));
    if hits.is_empty() {
        let msg = format!("No glossary entries match \"{}\".", query.trim());
        return html! { <p class="glossary-intro">{ msg }</p> };
    }
    let plural = if hits.len() == 1 { "match" } else { "matches" };
    let count = format!("{} {plural}, best first:", hits.len());
    let cards = hits
        .iter()
        .enumerate()
        .map(|(pos, &i)| entry_html(&entries[i], pos == 0));
    html! { <>
        <p class="glossary-intro">{ count }</p>
        { for cards }
    </> }
}

/// One glossary entry card, highlighted when it is the best match.
fn entry_html(e: &GlossaryEntry, best: bool) -> Html {
    let class = if best {
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
}

/// Search-box input handler: mirror the box into the query state.
fn query_input_callback(query: &UseStateHandle<String>) -> Callback<InputEvent> {
    let query = query.clone();
    Callback::from(move |e: InputEvent| {
        let target: HtmlInputElement = e.target_unchecked_into();
        query.set(target.value());
    })
}
