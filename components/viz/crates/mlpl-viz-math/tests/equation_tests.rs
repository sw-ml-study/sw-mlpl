//! `render_equation`: a Unicode math string -> a self-contained,
//! high-contrast SVG visual (the `svg(text, "equation")` render).

use mlpl_viz_math::render_equation;

#[test]
fn renders_a_self_contained_high_contrast_svg() {
    let svg = render_equation("y = m\u{00b7}x + b"); // middle dot
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
    assert!(svg.contains("fill=\"#1e1e2e\"")); // dark canvas
    assert!(svg.contains("fill=\"#cdd6f4\"")); // light glyphs
    assert!(svg.contains("<text"));
    assert!(svg.contains('\u{00b7}')); // the Unicode middle dot survives
}

#[test]
fn multiline_emits_one_text_element_per_line() {
    let svg = render_equation("line one\nline two\nline three");
    assert_eq!(svg.matches("<text").count(), 3);
}

#[test]
fn subscripts_and_superscripts_become_shifted_tspans() {
    let svg = render_equation("x_i + y^2 + z_{max}");
    assert!(svg.contains("baseline-shift=\"sub\""));
    assert!(svg.contains("baseline-shift=\"super\""));
    assert!(svg.contains(">max<")); // braced multi-char subscript
}

#[test]
fn xml_special_chars_are_escaped() {
    let svg = render_equation("a < b & c");
    assert!(svg.contains("&lt;"));
    assert!(svg.contains("&amp;"));
    assert!(!svg.contains("a < b")); // no raw '<'
}

#[test]
fn empty_input_is_still_a_valid_svg() {
    let svg = render_equation("");
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
}

#[test]
fn big_operator_limits_stack_above_and_below() {
    // A sum with limits renders the operator AND its bounds centered
    // (text-anchor middle), with the bounds as their own elements -- the
    // stacked/display convention, not inline to the right.
    let svg = render_equation("y = \u{2211}_{q=1}^{Q} x[q]");
    assert!(svg.contains("text-anchor=\"middle\""), "{svg}");
    assert!(svg.contains(">Q<"), "upper limit present: {svg}");
    assert!(svg.contains(">q=1<"), "lower limit present: {svg}");
    assert!(svg.contains('\u{2211}'), "the sum sign survives: {svg}");
}
