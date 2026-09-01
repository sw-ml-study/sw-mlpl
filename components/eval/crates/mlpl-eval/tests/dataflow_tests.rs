//! `dataflow(nodes, edges)` interpreter builtin: node/edge records ->
//! an SVG string (../emufpga ask 4, Phase 1). A visualization surface
//! like `svg`, so it returns a `Value::Str` holding the SVG.

use mlpl_eval::{Environment, Value, eval_program_value};

fn eval(src: &str) -> Value {
    let toks = mlpl_parser::lex(src).expect("lex");
    let stmts = mlpl_parser::parse(&toks).expect("parse");
    let mut env = Environment::new();
    eval_program_value(&stmts, &mut env).expect("eval")
}

#[test]
fn dataflow_returns_an_svg_string() {
    let v = eval(
        "nodes = {labels: [\"storage\", \"FIFO\", \"lanes\"]}\n\
         edges = {from: [0, 1], to: [1, 2], labels: [\"stream\", \"issue\"]}\n\
         dataflow(nodes, edges)\n",
    );
    let Value::Str(svg) = v else {
        panic!("dataflow should return a string, got {v:?}");
    };
    assert!(svg.starts_with("<svg") && svg.trim_end().ends_with("</svg>"));
    assert!(svg.contains("storage") && svg.contains("stream"));
    assert!(svg.contains("marker-end=\"url(#aw)\""));
}

#[test]
fn dataflow_edges_may_omit_labels() {
    let v = eval("dataflow({labels: [\"a\", \"b\"]}, {from: [0], to: [1]})\n");
    assert!(matches!(v, Value::Str(s) if s.starts_with("<svg")));
}

#[test]
fn dataflow_reports_a_shape_error_not_a_panic() {
    let toks = mlpl_parser::lex("dataflow({labels: [\"a\"]}, {from: [0], to: [5]})\n").unwrap();
    let stmts = mlpl_parser::parse(&toks).unwrap();
    let mut env = Environment::new();
    assert!(eval_program_value(&stmts, &mut env).is_err());
}
