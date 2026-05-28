//! Saga 21.5 step 005: viz-format detection table.
//!
//! `VizFormat::detect` recognizes SVG, HTML, PNG, JPEG (by magic
//! bytes / text-prefix sniffing) plus an explicit `application/json`
//! opt-in. Returns `None` when no format matches -- callers fall
//! back to "store as opaque bytes" (or, for the `mlpl-cli`
//! cache path, decline to write the file).
//!
//! Each variant exposes `content_type` (the MIME for `/v1/viz`
//! responses) and `extension` (used by `write_to_cache` to pick
//! the right `<hash>.<ext>` suffix).

use mlpl_cli::viz_cache::VizFormat;

// ---- SVG ----

#[test]
fn detects_bare_svg_payload() {
    assert_eq!(VizFormat::detect(b"<svg width='1'/>"), Some(VizFormat::Svg));
}

#[test]
fn detects_xml_prolog_then_svg() {
    let payload = b"<?xml version='1.0'?><svg></svg>";
    assert_eq!(VizFormat::detect(payload), Some(VizFormat::Svg));
}

// ---- HTML ----

#[test]
fn detects_doctype_html_payload() {
    let payload = b"<!DOCTYPE html><html><body>x</body></html>";
    assert_eq!(VizFormat::detect(payload), Some(VizFormat::Html));
}

#[test]
fn detects_bare_html_tag() {
    let payload = b"<html><body>hi</body></html>";
    assert_eq!(VizFormat::detect(payload), Some(VizFormat::Html));
}

#[test]
fn html_detection_ignores_leading_whitespace() {
    let payload = b"\n  <!DOCTYPE html>\n<html></html>";
    assert_eq!(VizFormat::detect(payload), Some(VizFormat::Html));
}

// ---- PNG ----

#[test]
fn detects_png_magic_bytes() {
    // Real PNG signature: 89 50 4E 47 0D 0A 1A 0A.
    let payload: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
    assert_eq!(VizFormat::detect(payload), Some(VizFormat::Png));
}

#[test]
fn rejects_almost_png_signature() {
    // First byte differs from 0x89.
    let payload: &[u8] = &[0x88, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    assert_ne!(VizFormat::detect(payload), Some(VizFormat::Png));
}

// ---- JPEG ----

#[test]
fn detects_jpeg_magic_bytes() {
    // JPEG/JFIF SOI: FF D8 FF (E0|E1|...) -- the third byte's
    // upper nibble is always F.
    let payload: &[u8] = &[0xFF, 0xD8, 0xFF, 0xE0, 0, 0];
    assert_eq!(VizFormat::detect(payload), Some(VizFormat::Jpeg));
}

// ---- JSON ----

#[test]
fn detects_json_when_opt_in_marker_present() {
    let payload = b"#mlpl-viz: json\n{\"loss\": [1.0, 0.5]}";
    assert_eq!(VizFormat::detect(payload), Some(VizFormat::Json));
}

#[test]
fn rejects_plain_json_without_opt_in() {
    // Plain JSON without the `#mlpl-viz: json` marker should not
    // false-positive as a viz: the eval pipeline often returns
    // string-valued primitives that happen to start with `{`.
    let payload = b"{\"loss\": [1.0, 0.5]}";
    assert_eq!(VizFormat::detect(payload), None);
}

// ---- No match ----

#[test]
fn returns_none_for_plain_text() {
    assert_eq!(VizFormat::detect(b"hello world"), None);
}

#[test]
fn returns_none_for_empty_input() {
    assert_eq!(VizFormat::detect(b""), None);
}

// ---- content_type / extension lookup ----

#[test]
fn content_types_match_expected_mime() {
    assert_eq!(VizFormat::Svg.content_type(), "image/svg+xml");
    assert_eq!(VizFormat::Html.content_type(), "text/html");
    assert_eq!(VizFormat::Png.content_type(), "image/png");
    assert_eq!(VizFormat::Jpeg.content_type(), "image/jpeg");
    assert_eq!(VizFormat::Json.content_type(), "application/json");
}

#[test]
fn extensions_match_expected_suffix() {
    assert_eq!(VizFormat::Svg.extension(), "svg");
    assert_eq!(VizFormat::Html.extension(), "html");
    assert_eq!(VizFormat::Png.extension(), "png");
    assert_eq!(VizFormat::Jpeg.extension(), "jpg");
    assert_eq!(VizFormat::Json.extension(), "json");
}

// ---- is_svg_string back-compat ----

#[test]
fn is_svg_string_still_recognizes_svg() {
    use mlpl_cli::viz_cache::is_svg_string;
    assert!(is_svg_string("<svg/>"));
    assert!(is_svg_string("<?xml ?><svg/>"));
    assert!(!is_svg_string("plain text"));
}
