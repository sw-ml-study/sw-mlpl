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

pub(crate) fn fmt_fn_def(
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

pub(crate) fn fmt_if_expr(
    f: &mut fmt::Formatter<'_>,
    cond: &Expr,
    then_body: &[Expr],
    else_body: &[Expr],
) -> fmt::Result {
    fmt_scope(f, &format_args!("if {cond}"), then_body)?;
    fmt_scope(f, &format_args!(" else"), else_body)
}

/// The two quoted-string atoms, split from the big `Display`
/// match for the function-LOC budget.
pub(crate) fn fmt_quoted(f: &mut fmt::Formatter<'_>, e: &Expr) -> fmt::Result {
    match e {
        Expr::StrLit(s, _) => {
            write!(f, "\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
        }
        Expr::Include(p, _) => write!(f, "include \"{p}\""),
        _ => unreachable!("fmt_quoted only receives StrLit/Include"),
    }
}
