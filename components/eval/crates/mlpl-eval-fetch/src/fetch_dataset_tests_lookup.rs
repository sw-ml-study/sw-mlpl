//! Dataset lookup-registry tests.

use super::*;

#[test]
fn lookup_known_dataset_returns_spec() {
    let s = lookup("oxford_iiit_pet").expect("known dataset");
    assert_eq!(s.subdir, "oxford-iiit-pet");
    assert_eq!(s.target_h, 128);
    assert_eq!(s.target_w, 128);
}

#[test]
fn lookup_unknown_dataset_errors() {
    let res = lookup("nope");
    assert!(res.is_err(), "expected unknown dataset error");
    if let Err(e) = res {
        let msg = format!("{e}");
        assert!(msg.contains("unknown dataset"), "got: {msg}");
    }
}
