# Typed Trace Events Contract (Saga 23 step 007)

## Purpose

Extend the `mlpl-trace` JSON schema with optional per-event
type fields so a traced run becomes a typed transformation log
rather than a shape-and-values log. A student reading the
trace JSON sees `softmax: Logit -> Probability` for every
typed assignment, not just `softmax: [4, 8] -> [4, 8]`.

This is the educational keystone of Saga 23: the trace export
is the canonical artifact, the web REPL trace panel and the
`--trace-json` file both consume it, and any future viewer can
render the typed transformations without re-running the
program.

## Schema additions

`TraceEvent` (in `crates/mlpl-trace/src/event.rs`) gains two
optional fields:

```rust
pub struct TraceEvent {
    pub seq: u64,
    pub op: String,
    pub span: Span,
    pub inputs: Vec<TraceValue>,
    pub output: TraceValue,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_types: Vec<Option<ValueTag>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_type: Option<ValueTag>,
}
```

`ValueTag` is the curated Tier A enum from `mlpl-core` (Saga
23 step 001). It serializes as serde-default tagged JSON: unit
variants render as the variant name string, struct variants
render as a single-key object.

Examples:

```json
{
  "seq": 4,
  "op": "assign",
  "span": { ... },
  "inputs": [...],
  "output": {...},
  "output_type": "Probability"
}

{
  "seq": 7,
  "op": "assign",
  "span": { ... },
  "inputs": [...],
  "output": {...},
  "input_types": ["Logit", null],
  "output_type": { "Loss": { "kind": "CrossEntropy" } }
}
```

## When the fields are populated

`input_types` and `output_type` are populated only for
assignment expressions (`Expr::Assign { name, value }`):

- `output_type` is set to whatever tag the binding carries
  after the assignment completes -- i.e. the result of the
  Saga 23 step 002/003 producer rules and step 005
  propagation rules.
- `input_types` is one entry per arg of the rhs FnCall,
  looked up by name when the arg is a bare identifier:
  `cross_entropy(L, T)` produces `[get_tag("L"),
  get_tag("T")]`. Non-identifier args (literals, nested
  FnCalls, BinOps) get `None`.

For all other expression kinds (literals, identifier reads,
binops, unary negation, fncalls, repeat / train bodies, etc.)
both fields stay at their default values (`Vec::new()` and
`None`) and the serde `skip_serializing_if` keeps them out of
the JSON entirely.

## Backwards compatibility

A program with no `set_tag` calls and no producer-tagged ops
serializes IDENTICALLY before and after step 007. The
`skip_serializing_if = "Vec::is_empty" / "Option::is_none"`
attributes ensure the new fields disappear from the JSON when
they would only carry default values.

Additionally, the helper that derives `input_types` skips the
vec entirely when *no* input carries a tag. So a typed-output
event with all-untagged inputs serializes as

```json
{ "op": "assign", "output_type": "Probability", ... }
```

(no `input_types` key) rather than

```json
{ "op": "assign", "input_types": [null, null], "output_type": "Probability", ... }
```

This keeps the JSON terse for the common case (the user has
adopted typed values for outputs but their input bindings are
not yet tagged).

## Round-trip guarantee

`Trace::to_json` followed by `serde_json::from_str::<Trace>`
preserves both fields including the metadata inside struct
variants: a `Loss { kind: CrossEntropy }` round-trips with
its `kind` intact, a `Gradient { wrt: "W1" }` round-trips
with the parameter name intact.

## Out of scope

- Special-case trace push sites (`grad`, `matmul`,
  `adam`/`momentum_sgd`) populate `input_types: []` and
  `output_type: None`. Adding type info there is a small
  follow-up; for the educational use case the assign-site
  events are the load-bearing surface (a student looks at
  `loss = cross_entropy(L, Y)` and sees `Loss(CrossEntropy)`,
  which is what the assign-site event carries).
- Inline expression types: `cross_entropy(softmax(L, 1), Y)`
  does not record the inner `softmax` step as its own typed
  event. The inline producer's tag flows into the predicate
  (Saga 23 step 004) but the trace records only the outer
  `assign` event. A richer per-op trace expansion would land
  in a future step.
- Human-readable trace formatter: there is no dedicated
  human-readable formatter today; the trace JSON is the
  output, consumed by the web REPL trace panel and any
  future viewer. The web REPL JS will pick up the new fields
  in a follow-up step.
- LayerRole, distribution kinds, ComputationGraph snapshots:
  these are Sagas 27 / 24 / 25 respectively and extend the
  TraceEvent schema with their own fields when they ship.

## Producer / consumer interaction

The trace's typed events are a *snapshot* of the side-table
state at event-push time. They do not run any producer or
predicate logic of their own:

- A `cross_entropy(softmax(L, 1), T)` that fires the Saga 23
  step 004 predicate raises a `TypeMismatch` BEFORE the
  trace event is pushed; no malformed event is written.
- A propagation error (`L + P` with mixed domains) similarly
  fails before the trace event lands.

The trace is the historical record of *what successfully
ran*, with the type metadata that was attached to each
binding when it ran. It is not a place for type errors or
hint messages; those belong in the user-facing error stream.
