//! `@test` interpretation at definition time: evaluate + validate
//! the metadata payload and register the source-ordered
//! `TestEntry` (docs/test-metadata-design.md). Other annotation
//! words are preserved data (the general namespace) and ignored
//! here.

use mlpl_core::Span;
use mlpl_parser::Expr;
use mlpl_trace::Trace;

use crate::env::Environment;
use mlpl_eval_state::TestEntry;
use mlpl_eval_types::{EvalError, Value};

/// Interpret a definition's annotations. Only `@test` registers;
/// duplicates by STABLE NAME are structured errors unless the
/// same function is being re-defined (replace in place).
pub(crate) fn register_def(
    fn_name: &str,
    annotations: &[(String, Option<Expr>)],
    span: &Span,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(), EvalError> {
    let Some((_, payload)) = annotations.iter().find(|(w, _)| w == "test") else {
        return Ok(());
    };
    let mut entry = TestEntry {
        name: fn_name.strip_prefix("u:").unwrap_or(fn_name).to_string(),
        fn_name: fn_name.to_string(),
        tags: Vec::new(),
        skip: String::new(),
        expected_failure: 0.0,
        timeout_ms: 0.0,
        source: env.current_source.clone().unwrap_or_else(|| "repl".into()),
        line: def_line(env, span),
    };
    if let Some(expr) = payload {
        apply_fields(&mut entry, expr, env, trace)?;
    }
    if let Some(prior) = env.tests.iter_mut().find(|t| t.name == entry.name) {
        if prior.fn_name != entry.fn_name {
            return Err(EvalError::Unsupported(format!(
                "@test: duplicate test name \"{}\" ({} in {}:{} vs {} in {}:{})",
                entry.name,
                prior.fn_name,
                prior.source,
                prior.line,
                entry.fn_name,
                entry.source,
                entry.line
            )));
        }
        *prior = entry; // re-definition keeps its order slot
        return Ok(());
    }
    env.tests.push(entry);
    Ok(())
}

/// 1-based line of the def within the currently evaluating text.
fn def_line(env: &Environment, span: &Span) -> usize {
    env.pending_source.as_ref().map_or(0, |text| {
        let upto = span.start.min(text.len());
        text[..upto].matches('\n').count() + 1
    })
}

/// Evaluate the `@test {...}` payload and fold its fields in.
/// Unknown fields are LOUD (malformed metadata must not pass).
fn apply_fields(
    entry: &mut TestEntry,
    payload: &Expr,
    env: &mut Environment,
    trace: &mut Option<&mut Trace>,
) -> Result<(), EvalError> {
    let Value::Record { fields } = crate::eval::eval_expr(payload, env, trace)? else {
        return Err(EvalError::Unsupported(
            "@test: the payload must be a record literal, e.g. @test {tags: [\"fast\"]}".into(),
        ));
    };
    for (key, value) in fields {
        match (key.as_str(), value) {
            ("name", Value::Str(s)) => entry.name = s,
            ("skip", Value::Str(s)) => entry.skip = s,
            ("tags", Value::StrList { items }) => entry.tags = items,
            ("expected_failure", Value::Array(a)) if a.rank() == 0 => {
                entry.expected_failure = a.data()[0];
            }
            ("timeout_ms", Value::Array(a)) if a.rank() == 0 => entry.timeout_ms = a.data()[0],
            (other, v) => {
                return Err(EvalError::Unsupported(format!(
                    "@test: unknown or mistyped field `{other}` ({}); recognized: name \
                     (string), tags (string list), skip (string), expected_failure \
                     (scalar), timeout_ms (scalar)",
                    mlpl_eval_types::value_kind(&v)
                )));
            }
        }
    }
    Ok(())
}
