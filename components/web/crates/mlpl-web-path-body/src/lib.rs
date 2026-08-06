//! Small markdown-ish renderer for learning-path step
//! bodies and tutorial-lesson intros.
//!
//! Handles only the constructs the existing body strings
//! use: blank-line paragraphs, "- " / "* " unordered lists,
//! "1. " / "2. " ordered lists, inline `code`, **bold**,
//! and _emph_. Not a full markdown parser -- the goal is
//! to stop rendering multi-paragraph bodies as one wall of
//! run-on text in the path-step panel.

mod block_acc;
mod blocks;
mod inline;
mod inline_render;
#[cfg(test)]
mod inline_tests;

use yew::prelude::*;

use blocks::Block;

/// Render `body` as a sequence of `<p>` / `<ul>` / `<ol>`
/// blocks. Adjacent bullet lines collapse into one list;
/// blank lines split paragraphs.
pub fn render_body(body: &str) -> Html {
    let parsed = blocks::parse_blocks(body);
    html! {
        <div class="path-body">
            { for parsed.into_iter().map(render_block) }
        </div>
    }
}

fn render_block(block: Block) -> Html {
    match block {
        Block::Paragraph(text) => html! {
            <p class="path-body-p">{ render_inline(&text) }</p>
        },
        Block::Unordered(items) => html! {
            <ul class="path-body-ul">
                { for items.into_iter().map(|t| html!{ <li>{ render_inline(&t) }</li> }) }
            </ul>
        },
        Block::Ordered(items) => html! {
            <ol class="path-body-ol">
                { for items.into_iter().map(|t| html!{ <li>{ render_inline(&t) }</li> }) }
            </ol>
        },
    }
}

/// Render a single line of body text as a sequence of inline
/// spans (text, `code`, **bold**, _emph_, `[[term]]`). No
/// block-level wrapping -- callers that want a paragraph use
/// `render_body` instead. Used by `entry_render` to put
/// glossary links inside MLPL `# comment` text.
pub fn render_inline(text: &str) -> Html {
    let parts = inline::split(text);
    html! { <>{ for parts.into_iter().map(inline_render::render_span) }</> }
}

/// Audit hook for prose corpora (demo intros/takeaways, lesson
/// bodies, glossary entries): the `_emph_` spans the inline
/// tokenizer would produce. Rendered prose in this project
/// never uses `_` emphasis intentionally, so a registry-wide
/// test pins this to empty -- any hit is an identifier being
/// eaten by markup (the snake_case class bug, 2026-08-06).
#[must_use]
pub fn emphasis_spans(text: &str) -> Vec<String> {
    inline::split(text)
        .into_iter()
        .filter_map(|s| match s {
            inline::Span::Emph(t) => Some(t),
            _ => None,
        })
        .collect()
}
