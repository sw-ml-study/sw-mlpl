//! Formatting helpers for compound AST nodes (multi-line
//! match arms in `Expr::fmt`). Keeps ast_fmt.rs under the
//! Function-LOC budget.

use std::fmt;

use crate::ast::Expr;

pub(crate) fn fmt_scope(
    f: &mut fmt::Formatter<'_>,
    head: &dyn fmt::Display,
    body: &[Expr],
) -> fmt::Result {
    write!(f, "{head} {{ ")?;
    for (i, e) in body.iter().enumerate() {
        if i > 0 {
            write!(f, "; ")?;
        }
        write!(f, "{e}")?;
    }
    write!(f, " }}")
}

pub(crate) fn fmt_record_lit(f: &mut fmt::Formatter<'_>, fields: &[(String, Expr)]) -> fmt::Result {
    write!(f, "{{")?;
    for (i, (name, value)) in fields.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{name}: {value}")?;
    }
    write!(f, "}}")
}

/// `{a, b: x} = value` -- a same-name binding prints as the bare field.
fn fmt_destructure(
    f: &mut fmt::Formatter<'_>,
    bindings: &[(String, String)],
    value: &Expr,
) -> fmt::Result {
    let parts: Vec<String> = bindings
        .iter()
        .map(|(field, var)| match field == var {
            true => field.clone(),
            false => format!("{field}: {var}"),
        })
        .collect();
    write!(f, "{{{}}} = {value}", parts.join(", "))
}

fn fmt_fn_def(
    f: &mut fmt::Formatter<'_>,
    name: &str,
    params: &[String],
    body: &[Expr],
) -> fmt::Result {
    write!(f, "def {name}(")?;
    for (i, p) in params.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{p}")?;
    }
    write!(f, ")")?;
    fmt_scope(f, &"", body)
}

/// The quoted-string atoms and the multi-field compound forms, split
/// from the big `Display` match for the function-LOC budget.
pub(crate) fn fmt_compound(f: &mut fmt::Formatter<'_>, e: &Expr) -> fmt::Result {
    match e {
        Expr::StrLit(s, _) => {
            write!(f, "\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
        }
        Expr::Include(p, _) => write!(f, "include \"{p}\""),
        Expr::Destructure {
            bindings, value, ..
        } => fmt_destructure(f, bindings, value),
        Expr::If {
            cond,
            then_body,
            else_body,
            ..
        } => {
            fmt_scope(f, &format_args!("if {cond}"), then_body)?;
            fmt_scope(f, &format_args!(" else"), else_body)
        }
        Expr::FnDef {
            name, params, body, ..
        } => fmt_fn_def(f, name, params, body),
        _ => unreachable!("fmt_compound only receives the forms routed to it"),
    }
}
