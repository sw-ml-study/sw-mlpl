Implement TraceEvent and Trace types in mlpl-trace with JSON serialization.

1. Add serde and serde_json as dependencies to mlpl-trace

2. In mlpl-trace, implement:
   - TraceEvent struct:
     - seq: u64 (sequence number)
     - op: String (operation description, e.g. "add", "reshape", "literal")
     - span: Span (source location)
     - inputs: Vec<TraceValue> (snapshots of input values)
     - output: TraceValue (result value)
   - TraceValue enum:
     - Scalar(f64)
     - Array { shape: Vec<usize>, data: Vec<f64> }
   - Trace struct:
     - events: Vec<TraceEvent>
     - source: String (original source code)
   - Trace::new(source) constructor
   - Trace::push(event) to add an event
   - Trace::to_json() -> String (serde serialization)

3. All types derive Serialize, Deserialize

TDD:
- Create a Trace, push events, serialize to JSON
- Deserialize JSON back to Trace, verify round-trip
- TraceValue from DenseArray conversion
- Empty trace serializes correctly

Allowed: crates/mlpl-trace/
May read: crates/mlpl-core/, crates/mlpl-array/
