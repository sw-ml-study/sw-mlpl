//! `Display` impls for AST nodes plus the shared rendering
//! helpers (`fmt_scope`, `fmt_comma_seq`, `fmt_record_lit`,
//! `write_seq`). Extracted from `ast.rs` so the AST module
//! itself only declares types + spans, keeping it under the
//! sw-checklist module-fn-count cap.
//!
//! Used by `device("mlx") { ... }` block forwarding (Saga R1
//! step 003) and by user-facing error messages that quote
//! source spans back to the caller.

use std::fmt;

use crate::ast::{BinOpKind, Expr, TensorCtorKind};

impl fmt::Display for BinOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for TensorCtorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Param => "param",
            Self::Tensor => "tensor",
        };
        write!(f, "{s}")
    }
}

fn fmt_comma_seq(f: &mut fmt::Formatter<'_>, exprs: &[Expr]) -> fmt::Result {
    for (i, e) in exprs.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{e}")?;
    }
    Ok(())
}

fn fmt_scope(f: &mut fmt::Formatter<'_>, head: &dyn fmt::Display, body: &[Expr]) -> fmt::Result {
    write!(f, "{head} {{ ")?;
    for (i, e) in body.iter().enumerate() {
        if i > 0 {
            write!(f, "; ")?;
        }
        write!(f, "{e}")?;
    }
    write!(f, " }}")
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IntLit(n, _) => write!(f, "{n}"),
            Self::FloatLit(x, _) if x.fract() == 0.0 && x.is_finite() => write!(f, "{x:.1}"),
            Self::FloatLit(x, _) => write!(f, "{x}"),
            Self::StrLit(s, _) => write!(f, "\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
            Self::Ident(name, _) => write!(f, "{name}"),
            Self::BuiltinRef(name, _) => write!(f, ":{name}"),
            Self::BinOp { op, lhs, rhs, .. } => write!(f, "({lhs} {op} {rhs})"),
            Self::UnaryNeg { operand, .. } => write!(f, "(-{operand})"),
            Self::Assign { name, value, .. } => write!(f, "{name} = {value}"),
            Self::RecordLit { fields, .. } => fmt_record_lit(f, fields),
            Self::FieldAccess {
                receiver, field, ..
            } => write!(f, "{receiver}.{field}"),
            Self::ArrayLit(elems, _) => write_seq(f, "[", "]", elems),
            Self::FnCall { name, args, .. } => write_seq(f, &format!("{name}("), ")", args),
            Self::TensorCtor { kind, shape, .. } => write_seq(f, &format!("{kind}["), "]", shape),
            Self::Repeat { count, body, .. } => fmt_scope(f, &format_args!("repeat {count}"), body),
            Self::Train { count, body, .. } => fmt_scope(f, &format_args!("train {count}"), body),
            Self::For {
                binding,
                source,
                body,
                ..
            } => fmt_scope(f, &format_args!("for {binding} in {source}"), body),
            Self::Experiment { name, body, .. } => {
                fmt_scope(f, &format_args!("experiment \"{name}\""), body)
            }
            Self::Device { target, body, .. } => {
                fmt_scope(f, &format_args!("device(\"{target}\")"), body)
            }
            Self::If {
                cond,
                then_body,
                else_body,
                ..
            } => {
                fmt_scope(f, &format_args!("if {cond}"), then_body)?;
                fmt_scope(f, &format_args!(" else"), else_body)
            }
            Self::While { cond, body, .. } => fmt_scope(f, &format_args!("while {cond}"), body),
            Self::Break { value: None, .. } => write!(f, "break"),
            Self::Break { value: Some(v), .. } => write!(f, "break {v}"),
            Self::Continue { .. } => write!(f, "continue"),
            Self::FnDef {
                name, params, body, ..
            } => {
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
            Self::Return { value: None, .. } => write!(f, "return"),
            Self::Return { value: Some(v), .. } => write!(f, "return {v}"),
        }
    }
}

fn fmt_record_lit(f: &mut fmt::Formatter<'_>, fields: &[(String, Expr)]) -> fmt::Result {
    write!(f, "{{")?;
    for (i, (name, value)) in fields.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{name}: {value}")?;
    }
    write!(f, "}}")
}

fn write_seq(f: &mut fmt::Formatter<'_>, open: &str, close: &str, items: &[Expr]) -> fmt::Result {
    write!(f, "{open}")?;
    fmt_comma_seq(f, items)?;
    write!(f, "{close}")
}
