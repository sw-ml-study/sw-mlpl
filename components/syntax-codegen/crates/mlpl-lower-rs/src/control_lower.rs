//! Lowering control-flow expressions (compiler-control-flow) plus the
//! `CVal`-return pre-pass that classifies user functions by flow.
//! Branch (and loop) bodies reuse `fndef_lower::lower_body`, so
//! `return` inside a branch emits a real Rust return. Truthiness
//! follows the interpreter: a non-zero scalar is true.

use std::collections::HashSet;

use mlpl_parser::Expr;
use proc_macro2::TokenStream;
use quote::quote;

use crate::{Ctx, LowerError, fndef_lower, lower_expr};

/// Pre-pass: the bare names of user functions whose body produces a
/// `CVal` -- it builds a record, calls `ok`/`err`, or uses `?`
/// (`check`) anywhere. Such functions lower to `-> CVal`. Run before
/// any body is lowered so a call to a function defined later still
/// resolves to the right return mode.
pub(crate) fn collect_cval_returning(stmts: &[Expr]) -> HashSet<String> {
    let mut out = HashSet::new();
    for s in stmts {
        if let Expr::FnDef { name, body, .. } = s
            && body.iter().any(expr_has_cval_marker)
        {
            out.insert(name.strip_prefix("u:").unwrap_or(name).to_string());
        }
    }
    out
}

/// Does this expression (recursively) build a record or call
/// `ok`/`err`/`check`? Marks a function body as `CVal`-returning.
fn expr_has_cval_marker(e: &Expr) -> bool {
    let any = |es: &[Expr]| es.iter().any(expr_has_cval_marker);
    match e {
        Expr::RecordLit { .. } => true,
        Expr::FnCall { name, args, .. } => {
            matches!(name.as_str(), "ok" | "err" | "check") || any(args)
        }
        Expr::Assign { value: e, .. }
        | Expr::UnaryNeg { operand: e, .. }
        | Expr::FieldAccess { receiver: e, .. }
        | Expr::Return { value: Some(e), .. } => expr_has_cval_marker(e),
        Expr::BinOp { lhs, rhs, .. } => expr_has_cval_marker(lhs) || expr_has_cval_marker(rhs),
        Expr::ArrayLit(elems, _) => any(elems),
        Expr::If {
            cond,
            then_body,
            else_body,
            ..
        } => expr_has_cval_marker(cond) || any(then_body) || any(else_body),
        Expr::While { cond, body, .. } => expr_has_cval_marker(cond) || any(body),
        _ => false,
    }
}

/// Lower `if cond { then } else { else }` to a Rust if-expression
/// over DenseArray truthiness (`cond.data()[0] != 0`). A branch that
/// diverges via `return` unifies with the other branch's value.
pub(crate) fn lower_if(
    ctx: &Ctx,
    cond: &Expr,
    then_body: &[Expr],
    else_body: &[Expr],
) -> Result<TokenStream, LowerError> {
    let c = lower_expr(ctx, cond)?;
    let t = fndef_lower::lower_body(ctx, then_body, false)?;
    let e = fndef_lower::lower_body(ctx, else_body, false)?;
    Ok(quote! { if (#c).data()[0] != 0.0 { #t } else { #e } })
}

/// Lower `while cond { body }`. The condition is re-evaluated each
/// iteration (its lowered form sits inline in the Rust `while`);
/// body assignments to variables declared before the loop reassign
/// them (mutation), so accumulators work. A `while` yields no value
/// -- the enclosing block yields a dummy scalar (discarded by the
/// statement position).
pub(crate) fn lower_while(
    ctx: &Ctx,
    cond: &Expr,
    body: &[Expr],
) -> Result<TokenStream, LowerError> {
    let c = lower_expr(ctx, cond)?;
    let b = fndef_lower::lower_body(ctx, body, false)?;
    let rt = &ctx.rt;
    Ok(quote! {
        {
            while (#c).data()[0] != 0.0 {
                let _ = #b;
            }
            #rt::DenseArray::from_scalar(0.0)
        }
    })
}
