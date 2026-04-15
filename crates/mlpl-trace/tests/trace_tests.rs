use mlpl_core::Span;
use mlpl_trace::{Trace, TraceEvent, TraceValue};

#[test]
fn empty_trace_json() {
    let trace = Trace::new("".into());
    let json = trace.to_json();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["events"], serde_json::json!([]));
    assert_eq!(parsed["source"], "");
}

#[test]
fn trace_push_and_serialize() {
    let mut trace = Trace::new("1 + 2".into());
    trace.push(TraceEvent {
        seq: 0,
        op: "literal".into(),
        span: Span::new(0, 1),
        inputs: vec![],
        output: TraceValue::scalar(1.0),
    });
    trace.push(TraceEvent {
        seq: 1,
        op: "literal".into(),
        span: Span::new(4, 5),
        inputs: vec![],
        output: TraceValue::scalar(2.0),
    });
    trace.push(TraceEvent {
        seq: 2,
        op: "add".into(),
        span: Span::new(0, 5),
        inputs: vec![TraceValue::scalar(1.0), TraceValue::scalar(2.0)],
        output: TraceValue::scalar(3.0),
    });
    let json = trace.to_json();
    assert!(json.contains("\"add\""));
    assert_eq!(trace.events().len(), 3);
}

#[test]
fn trace_roundtrip_json() {
    let mut trace = Trace::new("x = 42".into());
    trace.push(TraceEvent {
        seq: 0,
        op: "literal".into(),
        span: Span::new(4, 6),
        inputs: vec![],
        output: TraceValue::scalar(42.0),
    });
    let json = trace.to_json();
    let parsed: Trace = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.source(), "x = 42");
    assert_eq!(parsed.events().len(), 1);
    assert_eq!(parsed.events()[0].op, "literal");
}

#[test]
fn trace_value_array() {
    let val = TraceValue::Array {
        shape: vec![2, 3],
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        labels: None,
    };
    let json = serde_json::to_string(&val).unwrap();
    assert!(json.contains("\"shape\":[2,3]"));
    let parsed: TraceValue = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, val);
}

#[test]
fn trace_value_from_dense_array() {
    use mlpl_array::{DenseArray, Shape};
    let arr = DenseArray::new(Shape::new(vec![2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let val = TraceValue::from_array(&arr);
    match val {
        TraceValue::Array { shape, data, .. } => {
            assert_eq!(shape, vec![2, 2]);
            assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        }
        _ => panic!("expected Array variant"),
    }
}

#[test]
fn trace_value_from_scalar_array() {
    use mlpl_array::DenseArray;
    let arr = DenseArray::from_scalar(7.0);
    let val = TraceValue::from_array(&arr);
    assert_eq!(val, TraceValue::Scalar { value: 7.0 });
}
