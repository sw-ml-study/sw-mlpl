Saga 23: Typed ML values + typed traces. Ship the optional-typing keystone per
docs/milestone-typed-values.md and docs/optional-typing-design.md. Add the
ValueTag side-table mechanism on Environment, auto-tag producers (softmax /
sigmoid / log_softmax / linear / embed / attention / cross_entropy / mse /
kl_divergence / grad / cosine_schedule / linear_warmup / attention_weights /
model apply), predicate-check consumers (cross_entropy / nll / sample / top_k /
entropy / confusion_matrix / adam / momentum_sgd) with EvalError::TypeMismatch
{ op, expected, actual, hint } carrying tutoring messages, define a tag
propagation table for arithmetic / transpose / reshape / reductions, rewrite
:describe / :vars / :tags / :untag to consume metadata, extend the trace JSON
schema with input_types/output_types fields, ship a tutoring error catalog in
crates/mlpl-eval/src/type_errors.rs, add a "Typed ML Values" web REPL tutorial
lesson, write docs/using-typed-values.md, and tag the release. Goal ranking:
educational > correctness > utility > practicality > maintainability >
extensibility > performance. Performance is explicitly last; runtime tag-checking
on every op entry and re-walking metadata in :describe is acceptable. Untyped
programs continue to run unchanged. Mixed (typed + untyped) programs are normal.
Annotations land later in Saga 26; this saga ships side-table + auto-tag +
predicate + typed surfaces only.