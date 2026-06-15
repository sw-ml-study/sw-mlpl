//! The pure mixed-content rule behind connect-mode reachability: an https
//! page cannot fetch a plain-http, non-local connect server. (The
//! browser-only `connect_blocked_reason` wrapper is exercised by the live
//! UI; here we lock down the decision logic without a browser.)

use mlpl_web_eval::connect_guard::is_mixed_content_blocked;

#[test]
fn https_page_to_http_lan_server_is_blocked() {
    assert!(is_mixed_content_blocked(true, "http://192.168.1.5:6464"));
    assert!(is_mixed_content_blocked(true, "http://host:6464"));
}

#[test]
fn localhost_and_loopback_are_exempt() {
    assert!(!is_mixed_content_blocked(true, "http://127.0.0.1:6464"));
    assert!(!is_mixed_content_blocked(true, "http://localhost:6464"));
    assert!(!is_mixed_content_blocked(true, "http://[::1]:6464"));
}

#[test]
fn https_server_and_http_page_are_fine() {
    // https page -> https server: no mixed content.
    assert!(!is_mixed_content_blocked(true, "https://host:6464"));
    // http page (local server UI) -> http server: same scheme, fine.
    assert!(!is_mixed_content_blocked(false, "http://host:6464"));
}
