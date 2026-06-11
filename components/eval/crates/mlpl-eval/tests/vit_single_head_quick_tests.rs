//! `demos/vit_single_head_quick.mlpl` integration test.
//! Saga 29 step 009.
//!
//! Runs the trained ViT demo end-to-end and asserts that
//! training accuracy clears the 60% gate from
//! docs/milestone-vit.md. Marked `#[ignore]` so the default
//! `cargo test` finishes in seconds; run the heavy training
//! suite with `cargo test --test vit_single_head_quick_tests
//! -- --ignored`.

use std::fs;
use std::path::PathBuf;

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn demo_source() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../demos/vit_single_head_quick.mlpl");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
#[ignore = "100 adam steps on a 20-image batch; takes ~30-60s on CPU"]
fn trained_vit_clears_the_60_percent_accuracy_gate() {
    let src = demo_source();
    let tokens = lex(&src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    let mut env = Environment::new();
    let v = eval_program_value(&stmts, &mut env).expect("eval");
    let accuracy = match v {
        Value::Array(a) => {
            assert_eq!(a.rank(), 0, "final value should be a scalar accuracy");
            a.data()[0]
        }
        other => panic!("expected scalar accuracy, got {other:?}"),
    };
    assert!(
        accuracy >= 0.60,
        "training accuracy {accuracy} fell below the 60% gate from milestone-vit.md"
    );
}
