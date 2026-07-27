//! `--cors-allow` origin-list parsing: the flag accepts a
//! comma-separated list so a server can serve both a dev page
//! (trunk on :9957) and a static page host (:6466) cross-origin.

use mlpl_serve_core::router_layers::parse_cors_origins;

#[test]
fn single_origin_parses() {
    let o = parse_cors_origins("http://localhost:9957");
    assert_eq!(o.len(), 1);
    assert_eq!(o[0], "http://localhost:9957");
}

#[test]
fn comma_list_parses_all_origins() {
    let o = parse_cors_origins("http://localhost:9957,http://large12:6466,http://localhost:6466");
    assert_eq!(o.len(), 3);
    assert_eq!(o[1], "http://large12:6466");
}

#[test]
fn whitespace_around_entries_is_trimmed() {
    let o = parse_cors_origins("http://a:1, http://b:2 ,  http://c:3");
    let vals: Vec<&str> = o.iter().map(|h| h.to_str().unwrap()).collect();
    assert_eq!(vals, ["http://a:1", "http://b:2", "http://c:3"]);
}

#[test]
fn empty_entries_are_dropped() {
    let o = parse_cors_origins("http://a:1,,http://b:2,");
    assert_eq!(o.len(), 2);
}

#[test]
#[should_panic(expected = "--cors-allow")]
fn invalid_origin_panics_with_flag_name() {
    let _ = parse_cors_origins("http://ok:1,not a header value\u{7f}");
}
