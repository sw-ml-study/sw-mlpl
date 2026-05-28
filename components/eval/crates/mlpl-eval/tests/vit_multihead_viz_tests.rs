//! Integration tests for the multi-head ViT viz demos.
//! Saga 29 step 014.
//!
//! `demos/vit_attention_pattern_multihead.mlpl` is a no-
//! training forward pass and runs in milliseconds; we
//! exercise it inline. The trained companion
//! `demos/vit_multihead_quick.mlpl` runs 100 adam steps and
//! takes ~minute on CPU; it ships as a #[ignore]'d test.

use mlpl_eval::{Environment, eval_program_value};
use mlpl_parser::{lex, parse};

fn load_demo(path: &str) -> String {
    std::fs::read_to_string(path).expect("demo file readable")
}

fn run_program_returning_str(src: &str) -> String {
    let stmts = parse(&lex(src).unwrap()).unwrap();
    let mut env = Environment::new();
    let result = eval_program_value(&stmts, &mut env).unwrap();
    match result {
        mlpl_eval::Value::Str(s) => s,
        other => panic!("expected string final value, got {other:?}"),
    }
}

#[test]
fn vit_attention_pattern_multihead_demo_runs_end_to_end() {
    // Full evaluation including the trailing row_sums echo.
    // Asserting the final value is a tensor whose entries
    // are all 1 (softmax row sums per head).
    let demo = load_demo("../../demos/vit_attention_pattern_multihead.mlpl");
    let stmts = parse(&lex(&demo).unwrap()).unwrap();
    let mut env = Environment::new();
    let v = eval_program_value(&stmts, &mut env).unwrap();
    let arr = v.into_array().expect("row_sums is a tensor");
    assert_eq!(arr.shape().dims(), &[4, 17]);
    for (i, &x) in arr.data().iter().enumerate() {
        assert!(
            (x - 1.0).abs() < 1e-9,
            "row sum at index {i} is {x}, not 1.0"
        );
    }
}

#[test]
fn vit_attention_pattern_multihead_demo_emits_heatmap_grid_svg() {
    // Same demo but trimmed so the final statement is the
    // svg() call; we then check the SVG has four head labels.
    let demo = load_demo("../../demos/vit_attention_pattern_multihead.mlpl");
    let mut trimmed = String::new();
    for line in demo.lines() {
        trimmed.push_str(line);
        trimmed.push('\n');
        if line.trim_start().starts_with("svg(") {
            break;
        }
    }
    let svg = run_program_returning_str(&trimmed);
    assert!(
        svg.starts_with("<svg "),
        "not an SVG: {}",
        &svg[..80.min(svg.len())]
    );
    for h in 0..4 {
        let needle = format!("head {h}");
        assert!(
            svg.contains(&needle),
            "expected SVG to contain '{needle}'; got first 200 chars:\n{}",
            &svg[..200.min(svg.len())]
        );
    }
    // 4 heads x 17x17 cells = 4 + 4*17*17 = ~1160 <rect> elements.
    let rect_count = svg.matches("<rect").count();
    assert!(
        rect_count >= 4 * 17 * 17,
        "expected >= {} <rect> elements, found {rect_count}",
        4 * 17 * 17
    );
}

#[ignore = "100 adam steps on a 20-image batch + 4-head attention; takes ~60-90s on CPU"]
#[test]
fn vit_multihead_quick_demo_runs_end_to_end() {
    let demo = load_demo("../../demos/vit_multihead_quick.mlpl");
    let stmts = parse(&lex(&demo).unwrap()).unwrap();
    let mut env = Environment::new();
    let v = eval_program_value(&stmts, &mut env).unwrap();
    let acc = v.into_array().expect("accuracy is a scalar");
    assert_eq!(acc.shape().dims(), &[] as &[usize]);
    let val = acc.data()[0];
    // 4-head ViT on 20 cat/dog images for 100 steps should
    // clear 60% training accuracy with the seed pinned in
    // the demo. Match the threshold the single-head demo
    // uses so a regression in multi-head training shows up.
    assert!(
        val >= 0.6,
        "multi-head training accuracy {val} below 60% threshold"
    );
}
