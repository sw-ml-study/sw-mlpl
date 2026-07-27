//! `?connect=` value validation: malformed values (stray characters
//! from copy-paste, missing port, bad scheme) must fail fast with a
//! specific message instead of silently probing garbage.

use mlpl_web_eval_core::eval_url::validate_connect_url;

#[test]
fn accepts_the_canonical_forms() {
    for ok in [
        "http://large12:6464",
        "http://localhost:6464",
        "http://127.0.0.1:6464/",
        "https://example.com:8443",
        "http://[::1]:6464",
    ] {
        assert!(validate_connect_url(ok).is_ok(), "{ok} should validate");
    }
}

#[test]
fn rejects_missing_port() {
    let err = validate_connect_url("http://host").unwrap_err();
    assert!(err.contains("port"), "message should name the port: {err}");
}

#[test]
fn rejects_stray_close_paren_from_docs_copy_paste() {
    let err = validate_connect_url("http://localhost:6464)").unwrap_err();
    assert!(
        err.contains("6464)"),
        "message should show the bad port: {err}"
    );
}

#[test]
fn rejects_missing_or_wrong_scheme() {
    assert!(validate_connect_url("large12:6464").is_err());
    assert!(validate_connect_url("ftp://large12:6464").is_err());
    let err = validate_connect_url("large12:6464").unwrap_err();
    assert!(
        err.contains("http://"),
        "message should show the fix: {err}"
    );
}

#[test]
fn rejects_paths_and_bad_hosts() {
    assert!(validate_connect_url("http://host:6464/api").is_err());
    assert!(validate_connect_url("http://:6464").is_err());
    assert!(validate_connect_url("http://ho st:6464").is_err());
}
