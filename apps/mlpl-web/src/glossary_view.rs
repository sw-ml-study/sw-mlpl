//! Glossary tab content for the `?` documentation dialog.
//!
//! Parses `docs/glossary.md` (`include_str!`'d at compile time)
//! into per-term entries with stable id slugs, then renders a
//! search box plus a scrollable list. Typing in the search box
//! jumps the list to the first term whose first word matches
//! the typed prefix (case-insensitive). Educational-first: a
//! student looking for "MLP" types M, L, P and watches the
//! list scroll to MLP without a mouse round-trip.

use std::sync::OnceLock;

use wasm_bindgen::JsCast;
use web_sys::HtmlInputElement;
use yew::prelude::*;

const GLOSSARY_MD: &str = include_str!("../../../docs/glossary.md");

pub struct GlossaryEntry {
    pub term: String,
    pub slug: String,
    pub body: String,
}

pub struct GlossaryDoc {
    pub intro: String,
    pub entries: Vec<GlossaryEntry>,
}

fn doc() -> &'static GlossaryDoc {
    static DOC: OnceLock<GlossaryDoc> = OnceLock::new();
    DOC.get_or_init(parse_glossary)
}

fn parse_glossary() -> GlossaryDoc {
    let mut intro = String::new();
    let mut entries = Vec::new();
    let mut chunks = GLOSSARY_MD.split("\n## ");
    if let Some(first) = chunks.next() {
        intro = first
            .lines()
            .filter(|l| !l.starts_with("# ") && !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
    }
    for chunk in chunks {
        if let Some((term_line, rest)) = chunk.split_once('\n') {
            let term = term_line.trim().to_string();
            let slug = slugify(&term);
            entries.push(GlossaryEntry {
                term,
                slug,
                body: rest.trim().to_string(),
            });
        }
    }
    GlossaryDoc { intro, entries }
}

fn slugify(term: &str) -> String {
    let mut out = String::with_capacity(term.len());
    let mut last_dash = true;
    for c in term.chars() {
        if c.is_ascii_alphanumeric() {
            out.extend(c.to_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Find the first entry whose first whitespace-delimited word
/// starts with `query` (case-insensitive). Returns the entry's
/// index in `entries`, or `None` for an empty / unmatched
/// query.
fn find_match(query: &str, entries: &[GlossaryEntry]) -> Option<usize> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return None;
    }
    entries.iter().position(|e| {
        let first_word = e.term.split_whitespace().next().unwrap_or("");
        first_word.to_lowercase().starts_with(&q)
    })
}

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

    let matched = find_match(&query, &g.entries);
    let entries_html = g.entries.iter().enumerate().map(|(i, e)| {
        let class = if Some(i) == matched {
            "glossary-entry matched"
        } else {
            "glossary-entry"
        };
        html! {
            <div class={class} id={e.slug.clone()}>
                <h3>{ &e.term }</h3>
                <div class="glossary-body">{ &e.body }</div>
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
    let Some(idx) = find_match(query, entries) else {
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
