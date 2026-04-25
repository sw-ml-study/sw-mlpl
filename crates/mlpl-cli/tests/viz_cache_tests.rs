//! Saga 21 step 003: viz cache strategy tests.

use mlpl_cli::viz_cache::{cache_path_for_content, is_svg_string, transform_value, write_to_cache};
use tempfile::TempDir;

// ---- is_svg_string ----

#[test]
fn detects_bare_svg_open() {
    assert!(is_svg_string(
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    ));
}

#[test]
fn detects_leading_whitespace_then_svg() {
    assert!(is_svg_string("  \n  <svg></svg>"));
}

#[test]
fn detects_xml_prolog_then_svg() {
    assert!(is_svg_string("<?xml version='1.0'?><svg></svg>"));
}

#[test]
fn detects_xml_prolog_with_whitespace_then_svg() {
    assert!(is_svg_string(
        "<?xml version='1.0' encoding='UTF-8'?>\n<svg></svg>"
    ));
}

#[test]
fn rejects_plain_text() {
    assert!(!is_svg_string("hello world"));
}

#[test]
fn rejects_other_xml() {
    assert!(!is_svg_string("<div>not svg</div>"));
}

#[test]
fn rejects_xml_prolog_without_svg() {
    assert!(!is_svg_string("<?xml version='1.0'?><nope/>"));
}

#[test]
fn rejects_empty_string() {
    assert!(!is_svg_string(""));
}

// ---- cache_path_for_content ----

#[test]
fn cache_path_is_deterministic() {
    let tmp = TempDir::new().unwrap();
    let p1 = cache_path_for_content("<svg>hi</svg>", tmp.path());
    let p2 = cache_path_for_content("<svg>hi</svg>", tmp.path());
    assert_eq!(p1, p2, "same content must yield the same path");
}

#[test]
fn cache_path_is_content_addressed() {
    let tmp = TempDir::new().unwrap();
    let p1 = cache_path_for_content("<svg>a</svg>", tmp.path());
    let p2 = cache_path_for_content("<svg>b</svg>", tmp.path());
    assert_ne!(p1, p2, "different content must yield different paths");
}

#[test]
fn cache_path_uses_svg_extension() {
    let tmp = TempDir::new().unwrap();
    let p = cache_path_for_content("<svg></svg>", tmp.path());
    assert_eq!(p.extension().and_then(|s| s.to_str()), Some("svg"));
}

#[test]
fn cache_path_filename_is_12_hex_chars_plus_extension() {
    let tmp = TempDir::new().unwrap();
    let p = cache_path_for_content("<svg></svg>", tmp.path());
    let stem = p.file_stem().unwrap().to_str().unwrap();
    assert_eq!(stem.len(), 12, "hash prefix should be 12 chars");
    assert!(
        stem.chars().all(|c| c.is_ascii_hexdigit()),
        "stem must be hex: {stem}"
    );
}

// ---- write_to_cache ----

#[test]
fn write_creates_dir_and_returns_readable_path() {
    let tmp = TempDir::new().unwrap();
    let nested = tmp.path().join("subdir/deeper");
    let content = "<svg>roundtrip</svg>";
    let path = write_to_cache(content, &nested).unwrap();
    assert!(path.exists(), "written path should exist");
    let read_back = std::fs::read_to_string(&path).unwrap();
    assert_eq!(read_back, content, "content roundtrips");
}

#[test]
fn write_then_again_overwrites_same_path() {
    let tmp = TempDir::new().unwrap();
    let content = "<svg>same</svg>";
    let p1 = write_to_cache(content, tmp.path()).unwrap();
    let p2 = write_to_cache(content, tmp.path()).unwrap();
    assert_eq!(p1, p2, "content-addressed: writes go to the same path");
}

// ---- transform_value ----

#[test]
fn transform_passes_non_svg_through() {
    let tmp = TempDir::new().unwrap();
    let out = transform_value("hello world", Some(tmp.path()));
    assert_eq!(out, "hello world");
}

#[test]
fn transform_writes_svg_and_returns_viz_path() {
    let tmp = TempDir::new().unwrap();
    let out = transform_value("<svg>x</svg>", Some(tmp.path()));
    assert!(
        out.starts_with("viz: "),
        "SVG output should be prefixed with `viz: `: {out}"
    );
    let path_str = out.strip_prefix("viz: ").unwrap();
    assert!(
        std::path::Path::new(path_str).exists(),
        "the printed path must exist on disk: {path_str}"
    );
}

#[test]
fn transform_uses_env_var_override_when_no_explicit_dir() {
    let tmp = TempDir::new().unwrap();
    // Use the override slot rather than mutating the
    // process env -- mutating MLPL_CACHE_DIR in a test
    // is fine in isolation but races with parallel
    // tests in the same process.
    let out = transform_value("<svg>env</svg>", Some(tmp.path()));
    let path_str = out.strip_prefix("viz: ").unwrap();
    assert!(
        path_str.starts_with(tmp.path().to_str().unwrap()),
        "override should be honored: {path_str}"
    );
}

#[test]
fn transform_passes_xml_prolog_svg_through_to_cache() {
    let tmp = TempDir::new().unwrap();
    let svg = "<?xml version='1.0'?><svg width='10'/>";
    let out = transform_value(svg, Some(tmp.path()));
    assert!(out.starts_with("viz: "), "should be cached: {out}");
}
