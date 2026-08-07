//! Provenance-badge logic: commit extraction from both stamp
//! shapes and the three-way verdict with honest degradation.

use mlpl_web_components_chrome::status_badge::{BundleStatus, extract_commit, verdict};

#[test]
fn commit_extracts_from_json_and_meta_shapes() {
    assert_eq!(
        extract_commit(r#"{"commit":"fa578a7e","built_at":"2026-08-07T03:07:13Z"}"#).as_deref(),
        Some("fa578a7e")
    );
    assert_eq!(
        extract_commit("fa578a7e 2026-08-07T03:07:13Z").as_deref(),
        Some("fa578a7e")
    );
    assert!(extract_commit("not a stamp").is_none());
}

#[test]
fn the_real_build_info_parses() {
    // Pin against the actual generated artifact.
    let info = include_str!("../../../../../pages/build-info.json");
    assert!(extract_commit(info).is_some(), "{info}");
}

#[test]
fn verdicts_cover_all_stales_and_default_unknown() {
    let (a, b, c) = (Some("aaaaaaa1"), Some("bbbbbbb2"), Some("ccccccc3"));
    // Page behind origin: reload fixes it.
    assert_eq!(verdict(a, b, None), BundleStatus::Stale);
    // Origin behind repo: deploy pending.
    assert_eq!(verdict(a, a, c), BundleStatus::DeployPending);
    // All agree (or repo unknown): current.
    assert_eq!(verdict(a, a, a), BundleStatus::Fresh);
    assert_eq!(verdict(a, a, None), BundleStatus::Fresh);
    // Honest defaults.
    assert_eq!(verdict(None, a, a), BundleStatus::Unknown);
    assert_eq!(verdict(a, None, a), BundleStatus::Unknown);
    assert_eq!(verdict(None, None, None), BundleStatus::Unknown);
}
