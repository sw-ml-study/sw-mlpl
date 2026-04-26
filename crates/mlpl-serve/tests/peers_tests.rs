//! Saga R1 step 003: peer registry unit tests.
//! Moved out of `peers.rs` so the source module
//! stays under the 7-fn cap.

use mlpl_serve::peers::{build_registry, parse_peer_arg};

#[test]
fn parse_arg_happy() {
    let (d, u) = parse_peer_arg("mlx=http://localhost:6465").unwrap();
    assert_eq!(d, "mlx");
    assert_eq!(u, "http://localhost:6465");
}

#[test]
fn parse_arg_rejects_missing_equals() {
    assert!(parse_peer_arg("mlx").is_err());
}

#[test]
fn parse_arg_rejects_empty_device() {
    assert!(parse_peer_arg("=http://localhost:6465").is_err());
}

#[test]
fn parse_arg_rejects_empty_url() {
    assert!(parse_peer_arg("mlx=").is_err());
}

#[test]
fn build_accepts_loopback_127_0_0_1() {
    let r = build_registry(vec![("mlx".into(), "http://127.0.0.1:6465".into())], false).unwrap();
    assert!(r.contains_key("mlx"));
}

#[test]
fn build_accepts_loopback_localhost() {
    let r = build_registry(vec![("mlx".into(), "http://localhost:6465".into())], false).unwrap();
    assert!(r.contains_key("mlx"));
}

#[test]
fn build_accepts_loopback_ipv6() {
    let r = build_registry(vec![("mlx".into(), "http://[::1]:6465".into())], false).unwrap();
    assert!(r.contains_key("mlx"));
}

#[test]
fn build_rejects_non_loopback_without_insecure() {
    let err = build_registry(
        vec![("mlx".into(), "http://192.168.1.10:6465".into())],
        false,
    )
    .unwrap_err();
    assert!(
        err.contains("--insecure-peers"),
        "error should reference flag: {err}"
    );
}

#[test]
fn build_accepts_non_loopback_with_insecure() {
    let r = build_registry(
        vec![("mlx".into(), "http://192.168.1.10:6465".into())],
        true,
    )
    .unwrap();
    assert!(r.contains_key("mlx"));
}

#[test]
fn build_rejects_duplicate_device() {
    let err = build_registry(
        vec![
            ("mlx".into(), "http://127.0.0.1:6465".into()),
            ("mlx".into(), "http://127.0.0.1:6466".into()),
        ],
        false,
    )
    .unwrap_err();
    assert!(err.contains("duplicate"));
}
