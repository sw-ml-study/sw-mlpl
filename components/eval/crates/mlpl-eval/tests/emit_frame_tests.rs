//! `emit_frame(name, step, x)` (Game of Life saga step 4): no-op
//! without a sink, captured with one, returns its tensor.

use std::sync::{Arc, Mutex};

use mlpl_eval::{Environment, MetricSink, eval_program_value};
use mlpl_parser::{lex, parse};

#[derive(Debug, Default)]
struct CaptureSink {
    frames: Mutex<Vec<(String, usize, Vec<usize>, Vec<f64>)>>,
}

impl MetricSink for CaptureSink {
    fn emit(&self, _name: &str, _step: usize, _value: f64) {}
    fn emit_frame(&self, name: &str, step: usize, shape: &[usize], values: &[f64]) {
        self.frames
            .lock()
            .unwrap()
            .push((name.to_string(), step, shape.to_vec(), values.to_vec()));
    }
}

fn run(env: &mut Environment, src: &str) -> mlpl_eval::Value {
    let prog = parse(&lex(src).expect("lex")).expect("parse");
    eval_program_value(&prog, env).expect("eval")
}

#[test]
fn no_sink_is_noop_and_returns_x() {
    let mut env = Environment::new();
    let v = run(
        &mut env,
        "emit_frame(\"life\", 0, reshape(iota(4), [2, 2]))",
    );
    match v {
        mlpl_eval::Value::Array(a) => assert_eq!(a.shape().dims(), &[2, 2]),
        other => panic!("expected array, got {other:?}"),
    }
}

#[test]
fn sink_captures_each_loop_frame() {
    let sink = Arc::new(CaptureSink::default());
    let mut env = Environment::new();
    env.set_metric_sink(sink.clone());
    run(
        &mut env,
        "g = reshape(iota(4), [2, 2]); i = 0; while lt(i, 3) { emit_frame(\"life\", i, g); i = i + 1 }",
    );
    let frames = sink.frames.lock().unwrap();
    assert_eq!(frames.len(), 3);
    assert_eq!(frames[0].0, "life");
    assert_eq!(frames[2].1, 2);
    assert_eq!(frames[0].2, vec![2, 2]);
    assert_eq!(frames[0].3, vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn wrong_arity_errors() {
    let mut env = Environment::new();
    let prog = parse(&lex("emit_frame(\"life\", 1)").unwrap()).unwrap();
    assert!(eval_program_value(&prog, &mut env).is_err());
}
