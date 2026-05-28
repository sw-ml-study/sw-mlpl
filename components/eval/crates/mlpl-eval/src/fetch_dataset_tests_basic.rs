//! Basic non-archive tests for fetch_dataset.

use std::fs;

use super::_test_helpers::temp_dir;
use super::*;

#[test]
fn label_for_assigns_cat_zero_dog_one() {
    assert_eq!(label_for("Abyssinian_1.jpg"), 0);
    assert_eq!(label_for("beagle_3.jpg"), 1);
    assert_eq!(label_for("12345.jpg"), 255);
}

#[test]
fn sha256_of_matches_known_content() {
    let tmp = temp_dir("sha256");
    let p = tmp.join("hello.txt");
    fs::write(&p, b"hello").unwrap();
    assert_eq!(
        crate::fetch_io::sha256_of(&p).unwrap(),
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
}

#[test]
fn ensure_tarball_passes_when_existing_hash_matches() {
    let tmp = temp_dir("ensure-ok");
    let p = tmp.join("blob");
    fs::write(&p, b"abc").unwrap();
    let want = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    crate::fetch_io::ensure_tarball(&p, "http://invalid.example/never", want).unwrap();
}

#[test]
fn ensure_tarball_errors_on_stale_hash() {
    let tmp = temp_dir("ensure-stale");
    let p = tmp.join("blob");
    fs::write(&p, b"abc").unwrap();
    let err = crate::fetch_io::ensure_tarball(&p, "http://invalid.example/never", "deadbeef")
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("sha256 mismatch"), "got: {msg}");
}
