Saga 23 step 007: typed-trace-events.

Extend the trace JSON schema to carry ValueTags so a traced run is no longer just shapes-and-values but a typed transformation log: 'softmax: Logit[batch=4, vocab=8] -> Probability[batch=4, vocab=8]' for every event. This is the educational keystone of the saga -- the trace export becomes a microscope on what each op *does* in semantic terms.

Rules to ship:

1. Extend mlpl-trace TraceEvent schema. Add optional fields:
   - input_types: Vec<Option<String>> (one entry per input; tag display_name + metadata as JSON-friendly representation, or null when untagged)
   - output_type: Option<String> (single output's tag, or null)
   These fields are 'omit when None' via serde, so existing trace JSON consumers see no change for untagged events.

2. Trace event production at the assignment site (eval.rs Expr::Assign): after computing the result tag (auto_tag::for_assign or tag_propagate::propagate), include it in the TraceEvent::output_type.

3. Trace event production for each fncall: when a producer rule fires, the matching tag goes into output_type. The eval_fncall path needs to pass tag info into the trace push.

4. Trace formatter: extend the human-readable trace formatter (the one used by --trace and the web REPL trace panel) to render the new shape:
   'softmax: Logit[batch=4, vocab=8] -> Probability[batch=4, vocab=8]'
   when types are present; fall back to the pre-step-007 format when both sides are untagged.

5. Trace JSON contract doc at contracts/eval-contract/typed-trace.md documenting the schema.

TDD: failing tests in crates/mlpl-trace/tests/typed_event_tests.rs and crates/mlpl-eval/tests/typed_trace_tests.rs (the eval-side tests should run a traced program and assert the trace JSON has the right input_types / output_type fields).

Implementation:
- mlpl-trace gains a dependency on mlpl-core for ValueTag (via re-export of a tag-name + serde-friendly Tag struct, OR have eval pass strings into the trace event so mlpl-trace stays decoupled).
- The trace event struct needs the new fields plus serde rename_field('skip_serializing_if = Option::is_none') so untagged events serialize unchanged.

Constraints:
- Existing trace JSON consumers (the web REPL trace panel, the --trace-json out file format) must round-trip unchanged for untagged programs.
- Run cargo test workspace-wide; the trace JSON shape test is the most likely place for surprise regressions.

Quality gates: full /mw-cp pass + markdown-checker on the new contract doc. sw-checklist failed count must hold at 139.