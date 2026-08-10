//! The picker uses explicit CSS tooltips rather than browser-dependent
//! native `title` popups, including wrappers around disabled buttons.

const MODE_BAR: &str = include_str!("../src/mode_bar.rs");
const WEB_HTML: &str = include_str!("../../../../web/crates/mlpl-web/index.html");

#[test]
fn picker_emits_visible_tooltip_targets() {
    assert!(MODE_BAR.contains("demo-tooltip-target"));
    assert!(MODE_BAR.matches("data-tooltip=").count() >= 2);
    assert!(MODE_BAR.contains("tabindex={if *disabled { \"0\" } else { \"-1\" }}"));
}

#[test]
fn tooltip_css_supports_hover_and_keyboard_focus() {
    assert!(WEB_HTML.contains(".demo-tooltip-target::after"));
    assert!(WEB_HTML.contains(".demo-tooltip-target:hover::after"));
    assert!(WEB_HTML.contains(".demo-tooltip-target:focus::after"));
    assert!(WEB_HTML.contains("pointer-events: none"));
}

#[test]
fn tooltip_is_offset_distinct_and_deliberately_delayed() {
    assert!(WEB_HTML.contains("left: calc(100% + 12px)"));
    assert!(WEB_HTML.contains("background: var(--crust)"));
    assert!(WEB_HTML.contains("border-color: var(--peach)"));
    assert!(WEB_HTML.contains("transition-delay: 250ms"));
}
