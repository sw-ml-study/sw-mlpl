//! Integration tests for `demos/vit_multihead_thorough.mlpl`.
//! Saga 29 step 015.
//!
//! Heavy: 200 adam steps on a 20-image batch with multi-head
//! attention. CPU-only run takes ~3-5 minutes; MLX cuts the
//! forward time but the backward (CPU-resident tape) still
//! dominates wall time on a tiny model.
//!
//! Both tests are `#[ignore]`'d so the default workspace test
//! pass stays fast. Run with
//!   `cargo test --release -p mlpl-eval --test vit_multihead_thorough_tests -- --ignored`
//! to validate the demo end-to-end.

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};

fn load_demo() -> String {
    std::fs::read_to_string("../../demos/vit_multihead_thorough.mlpl")
        .expect("vit_multihead_thorough.mlpl readable")
}

#[ignore = "200 adam steps on 20-image batch with multi-head attention; ~3-5 min CPU"]
#[test]
fn thorough_multi_head_clears_70_percent_accuracy_gate() {
    // 200 steps with 4 heads on a 20-image cat/dog set; with the
    // seed pinned in the demo we expect well over 70% training
    // accuracy. The single-head 100-step demo already hits 100%
    // on this same data; multi-head should easily clear the bar.
    let demo = load_demo();
    let stmts = parse(&lex(&demo).unwrap()).unwrap();
    let mut env = Environment::new();
    let v = eval_program_value(&stmts, &mut env).unwrap();
    let acc = v.into_array().expect("accuracy is a scalar");
    let val = acc.data()[0];
    assert!(
        val >= 0.7,
        "multi-head training accuracy {val} below 70% threshold"
    );
}
