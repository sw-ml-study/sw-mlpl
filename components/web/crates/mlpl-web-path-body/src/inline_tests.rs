//! Unit tests for the inline-span tokenizer (moved out of
//! `inline.rs` per the module-function-count budget).

use super::inline::{Span, split};

#[test]
fn glossary_sigil_round_trips() {
    // Core: [[term]] -> Span::Glossary("term"). Also
    // covers leading text + trailing text segmenting.
    let spans = split("see [[take (builtin)]] for more");
    assert_eq!(spans.len(), 3);
    assert!(matches!(&spans[0], Span::Text(t) if t == "see "));
    assert!(matches!(&spans[1], Span::Glossary(t) if t == "take (builtin)"));
    assert!(matches!(&spans[2], Span::Text(t) if t == " for more"));
}

#[test]
fn markdown_link_parses() {
    // [label](url) -> Span::Link; surrounding text segments.
    let spans = split("see [the doc](literate/x.html) now");
    assert!(matches!(&spans[0], Span::Text(t) if t == "see "));
    assert!(
        matches!(&spans[1], Span::Link { text, url } if text == "the doc" && url == "literate/x.html")
    );
    assert!(matches!(&spans[2], Span::Text(t) if t == " now"));
    // A bare [bracketed] with no (url) stays literal text, as
    // do a single-bracket "[term]" and an unterminated "[[".
    assert!(matches!(&split("[just text]")[0], Span::Text(_)));
    assert!(matches!(&split("[not a link]")[0], Span::Text(_)));
    assert!(matches!(&split("[[unterminated")[0], Span::Text(_)));
}

#[test]
fn intraword_underscores_stay_literal() {
    // The snake_case class bug (user report, 2026-08-06):
    // identifiers in prose must survive verbatim -- a `_`
    // inside a word is NEVER an emphasis delimiter.
    let text = "transpose_axes swaps axes and u:sudoku_candidate_safe_from checks peers";
    let spans = split(text);
    assert_eq!(spans.len(), 1, "{spans:?}");
    assert!(matches!(&spans[0], Span::Text(t) if t == text));
}

#[test]
fn word_boundary_emphasis_still_works() {
    let spans = split("this is _really_ important");
    assert!(
        matches!(&spans[1], Span::Emph(t) if t == "really"),
        "{spans:?}"
    );
    // Unclosed opener stays literal.
    assert!(matches!(&split("a _dangling tail")[0], Span::Text(_)));
}

#[test]
fn bracketed_prose_with_spaced_parenthetical_is_not_a_link() {
    // "[a, b] (note)" -- space between ] and ( already
    // prevents a link; a ](-adjacent pair whose url holds
    // whitespace must ALSO stay literal prose.
    let text = "[block-row, inner-col](APL2's 1 3 2 4 transpose, 0-based)";
    let spans = split(text);
    assert!(
        matches!(&spans[0], Span::Text(t) if t == text),
        "spaced url must not link: {spans:?}"
    );
}

#[test]
fn code_and_glossary_coexist() {
    // The dispatcher must not confuse `code` with [[term]].
    let spans = split("`code` and [[Stack (tape op)]]");
    assert!(matches!(&spans[0], Span::Code(t) if t == "code"));
    assert!(matches!(&spans[2], Span::Glossary(t) if t == "Stack (tape op)"));
}

#[test]
fn unicode_text_survives_tokenizing() {
    // APL glyphs in narration (docs/apl2-idioms.mlpl) shattered
    // when text accumulated BYTE by byte -- multi-byte UTF-8
    // must round-trip exactly.
    let text = "APL2: \u{2373}9 and 3 3\u{2374}\u{2373}9 with \u{2395}IO";
    let spans = split(text);
    assert_eq!(spans.len(), 1, "{spans:?}");
    assert!(matches!(&spans[0], Span::Text(t) if t == text), "{spans:?}");
    // Mixed with real markup, the unicode segments stay intact.
    let spans = split("dyadic \u{2349} maps to `transpose_axes`");
    assert!(matches!(&spans[0], Span::Text(t) if t == "dyadic \u{2349} maps to "));
    assert!(matches!(&spans[1], Span::Code(t) if t == "transpose_axes"));
}
