//! Tests for the `intro_md` inline renderer, kept in a sibling file so the
//! production module stays under the module-function-count budget.

use crate::intro_md::inline;

#[test]
fn boundary_star_is_emphasis() {
    assert_eq!(inline("a *word* b"), "a <em>word</em> b");
    assert_eq!(inline("*lead* then"), "<em>lead</em> then");
}

#[test]
fn midword_star_stays_literal() {
    // Quant-name wildcards (Q*_0, Q*_K, IQ*) are mid-word -- not italics.
    assert_eq!(
        inline("Q*_0 / Q*_K and IQ* names"),
        "Q*_0 / Q*_K and IQ* names"
    );
}

#[test]
fn double_star_is_bold() {
    assert_eq!(inline("a **word** b"), "a <strong>word</strong> b");
}

#[test]
fn code_span_keeps_stars_literal() {
    assert_eq!(inline("`a * b`"), "<code>a * b</code>");
}
