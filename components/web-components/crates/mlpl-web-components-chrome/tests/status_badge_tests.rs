//! Pure logic behind the header staleness badge.

use mlpl_web_components_chrome::status_badge::{BundleStatus, extract_bundle, verdict};

#[test]
fn bundle_name_extracts_from_index_and_script_urls() {
    let html = r#"<script type="module">import init from '/sw-mlpl/mlpl-web-5732a6778cd461f8.js';"#;
    assert_eq!(
        extract_bundle(html).as_deref(),
        Some("mlpl-web-5732a6778cd461f8")
    );
    assert_eq!(
        extract_bundle("https://host/sw-mlpl/mlpl-web-abcdef0123456789_bg.wasm").as_deref(),
        Some("mlpl-web-abcdef0123456789")
    );
    // Too-short hex (a stray mention) is not a bundle name.
    assert!(extract_bundle("mlpl-web-page is nice").is_none());
    assert!(extract_bundle("no bundle here").is_none());
}

#[test]
fn verdict_is_unknown_by_default_and_honest_offline() {
    assert_eq!(verdict(None, None), BundleStatus::Unknown);
    assert_eq!(
        verdict(Some("mlpl-web-aa11aa11"), None),
        BundleStatus::Unknown
    );
    assert_eq!(
        verdict(None, Some("mlpl-web-aa11aa11")),
        BundleStatus::Unknown
    );
}

#[test]
fn verdict_compares_bundles() {
    let a = Some("mlpl-web-aa11aa11");
    let b = Some("mlpl-web-bb22bb22");
    assert_eq!(verdict(a, a), BundleStatus::Fresh);
    assert_eq!(verdict(a, b), BundleStatus::Stale);
}
