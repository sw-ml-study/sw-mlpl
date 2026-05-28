//! `fetch_dataset(name)` builtin dispatch tests. Saga 29
//! step 004.
//!
//! Two flavors of test live here:
//!
//! - **Always-on dispatch tests** verify the arity / arg-type
//!   error paths and the registry's unknown-name path. They
//!   run regardless of the `image-io` feature so we get
//!   regression coverage on every CI build.
//! - **`#[ignore]` live-download integration** exercises the
//!   real Oxford-IIIT Pet tarball. It is gated on the
//!   `image-io` feature, runs only when `cargo test
//!   --ignored` is invoked, and assumes `MLPL_DATA_DIR`
//!   points at a checkout (or that the path is empty so the
//!   ~792 MB download lands fresh).

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};

#[test]
fn fetch_dataset_arity_error() {
    let tokens = lex("fetch_dataset()").unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let err = eval_program_value(&stmts, &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("fetch_dataset"), "got: {msg}");
}

#[test]
fn fetch_dataset_argument_must_be_a_string_literal() {
    let tokens = lex("fetch_dataset(42)").unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let err = eval_program_value(&stmts, &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("string literal") || msg.contains("disabled"),
        "got: {msg}"
    );
}

#[test]
fn fetch_dataset_unknown_name_errors_cleanly() {
    let tokens = lex("fetch_dataset(\"definitely_not_a_dataset\")").unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    // When image-io is on, lookup errors with "unknown
    // dataset". When it's off, the WASM-side stub errors
    // with "disabled in this build". Either message is a
    // tutoring error -- this assertion just checks we don't
    // panic.
    let err = eval_program_value(&stmts, &mut env).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("fetch_dataset") || msg.contains("disabled"),
        "got: {msg}"
    );
}

#[cfg(feature = "image-io")]
mod with_image_io {
    use super::*;
    use mlpl_eval::Value;

    /// Live tarball + extract + decode. Heavyweight; expects
    /// `MLPL_DATA_DIR` to point at the gitignored
    /// `data/oxford-iiit-pet/` checkout that already has
    /// `images.tar.gz` + `images/` from Saga 29 step 003.
    /// Run with `cargo test --features mlpl-eval/image-io
    /// -p mlpl-eval -- --ignored`.
    #[test]
    #[ignore]
    fn live_oxford_iiit_pet_returns_record_at_128_resolution() {
        // The decoded tensor is ~2.7 GB of f64 at 128x128.
        // Don't run unless the data dir is set up.
        if std::env::var("MLPL_DATA_DIR").is_err() {
            panic!(
                "set MLPL_DATA_DIR before running this --ignored test, \
                 e.g. MLPL_DATA_DIR=$(pwd)/data"
            );
        }
        let tokens = lex("fetch_dataset(\"oxford_iiit_pet\")").unwrap();
        let stmts = parse(&tokens).unwrap();
        let mut env = Environment::new();
        let v = eval_program_value(&stmts, &mut env).expect("fetch_dataset live");
        let Value::Record { fields } = v else {
            panic!("expected Record");
        };
        let x = match fields.get("X") {
            Some(Value::Array(a)) => a,
            _ => panic!("X missing"),
        };
        let dims = x.shape().dims();
        assert_eq!(dims.len(), 4, "X should be 4D, got {dims:?}");
        assert_eq!(dims[1], 3);
        assert_eq!(dims[2], 128);
        assert_eq!(dims[3], 128);
        assert!(
            dims[0] >= 7000,
            "Oxford-IIIT Pet has ~7393 images, got N={}",
            dims[0]
        );
        // Axis labels are the same as load_images.
        let labels = x.labels().expect("X must carry axis labels");
        let names: Vec<&str> = labels.iter().map(|l| l.as_deref().unwrap_or("")).collect();
        assert_eq!(names, vec!["batch", "channel", "y", "x"]);
        // Y has the same N.
        let y = match fields.get("Y") {
            Some(Value::Array(a)) => a,
            _ => panic!("Y missing"),
        };
        assert_eq!(y.shape().dims(), &[dims[0]]);
        // Names has the same N.
        let names_list = match fields.get("names") {
            Some(Value::StrList { items }) => items,
            _ => panic!("names missing"),
        };
        assert_eq!(names_list.len(), dims[0]);
    }
}
