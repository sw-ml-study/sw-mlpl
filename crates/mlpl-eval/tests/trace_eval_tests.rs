use mlpl_eval::{Environment, eval_program, eval_program_traced};
use mlpl_parser::{lex, parse};
use mlpl_trace::Trace;

fn traced(src: &str) -> Trace {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let mut trace = Trace::new(src.into());
    eval_program_traced(&stmts, &mut env, &mut trace).unwrap();
    trace
}

#[test]
fn trace_add_has_events() {
    let trace = traced("1 + 2");
    // Should have: literal(1), literal(2), add
    assert!(
        trace.events().len() >= 3,
        "got {} events",
        trace.events().len()
    );
    let ops: Vec<&str> = trace.events().iter().map(|e| e.op.as_str()).collect();
    assert!(ops.contains(&"literal"), "missing literal: {ops:?}");
    assert!(ops.contains(&"add"), "missing add: {ops:?}");
}

#[test]
fn trace_assign_array() {
    let trace = traced("x = [1, 2, 3]");
    let ops: Vec<&str> = trace.events().iter().map(|e| e.op.as_str()).collect();
    assert!(ops.contains(&"array_lit"), "missing array_lit: {ops:?}");
    assert!(ops.contains(&"assign"), "missing assign: {ops:?}");
}

#[test]
fn trace_fncall() {
    let trace = traced("iota(5)");
    let ops: Vec<&str> = trace.events().iter().map(|e| e.op.as_str()).collect();
    assert!(ops.contains(&"literal"), "missing literal: {ops:?}");
    assert!(ops.contains(&"fncall"), "missing fncall: {ops:?}");
}

#[test]
fn trace_seq_numbers_increment() {
    let trace = traced("1 + 2");
    let seqs: Vec<u64> = trace.events().iter().map(|e| e.seq).collect();
    for i in 1..seqs.len() {
        assert!(seqs[i] > seqs[i - 1], "seq not increasing: {seqs:?}");
    }
}

#[test]
fn trace_spans_valid() {
    let trace = traced("1 + 2");
    for event in trace.events() {
        assert!(event.span.start <= event.span.end);
    }
}

#[test]
fn trace_json_not_empty() {
    let trace = traced("1 + 2");
    let json = trace.to_json();
    assert!(json.contains("\"add\""));
    assert!(json.contains("\"literal\""));
}

#[test]
fn eval_without_trace_still_works() {
    let tokens = lex("1 + 2").unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    let arr = eval_program(&stmts, &mut env).unwrap();
    assert_eq!(arr.data(), &[3.0]);
}

// -- Label propagation through trace export (Saga 11.5 Phase 5) --

#[test]
fn trace_includes_axis_labels_on_labeled_array() {
    // An annotated assignment pushes a labeled array through the
    // trace. The JSON surface must carry those labels so downstream
    // tooling can reason about shape flow.
    let trace = traced("x : [seq, d] = reshape(iota(6), [2, 3])");
    let json = trace.to_json();
    assert!(
        json.contains("\"seq\""),
        "missing \"seq\" in trace:\n{json}"
    );
    assert!(json.contains("\"d\""), "missing \"d\" in trace:\n{json}");
}

#[test]
fn trace_unlabeled_array_has_no_labels_field() {
    // Labels are optional; unlabeled arrays must not emit stray label
    // metadata (keeps the trace JSON terse for the common case).
    let trace = traced("x = reshape(iota(6), [2, 3])");
    let json = trace.to_json();
    assert!(
        !json.contains("\"labels\""),
        "trace should omit labels on unlabeled arrays:\n{json}"
    );
}
