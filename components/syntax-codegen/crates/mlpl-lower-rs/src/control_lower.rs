//! Lowering control-flow expressions (compiler-control-flow) plus the
//! `CVal`-return pre-pass that classifies user functions by flow.
//! Branch (and loop) bodies reuse `fndef_lower::lower_body`, so
//! `return` inside a branch emits a real Rust return. Truthiness
//! follows the interpreter: a non-zero scalar is true.

use std::collections::HashSet;

use mlpl_parser::{BinOpKind, Expr};
use proc_macro2::TokenStream;
use quote::quote;

use crate::{Ctx, LowerError, fndef_lower, lower_darr, lower_expr};

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
        | Expr::Destructure { value: e, .. }
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

/// Lower an infix operator to an elementwise `apply_binop` (scalar
/// broadcasting). `and` / `or` keep the interpreter's control flow: a
/// scalar left side that decides the result short-circuits -- the right
/// side is not evaluated -- else the result is the 0/1 mask.
pub(crate) fn lower_binop(
    ctx: &Ctx,
    op: &BinOpKind,
    lhs: &Expr,
    rhs: &Expr,
) -> Result<TokenStream, LowerError> {
    let (l, r, rt) = (lower_darr(ctx, lhs)?, lower_darr(ctx, rhs)?, &ctx.rt);
    let closure = binop_closure(op);
    // UFCS through the runtime facade's re-exported trait, so the
    // generated call site needs no `use ApplyBinopExt`.
    let apply = |l: TokenStream| quote! { #rt::ApplyBinopExt::apply_binop(&(#l), &(#r), #closure).unwrap() };
    let decides = match op {
        BinOpKind::And => false,
        BinOpKind::Or => true,
        _ => return Ok(apply(l)),
    };
    let masked = apply(quote! { __l });
    Ok(quote! {{
        let __l = #l;
        if __l.rank() == 0 && (__l.data()[0] != 0.0) == #decides {
            #rt::DenseArray::from_scalar(if #decides { 1.0 } else { 0.0 })
        } else {
            #masked
        }
    }})
}

/// The elementwise `f64` closure for an operator; comparisons and the
/// logical operators yield 0/1 (`eq` within `f64::EPSILON`).
fn binop_closure(op: &BinOpKind) -> TokenStream {
    let cmp = |t: TokenStream| quote! { |__a: f64, __b: f64| if #t { 1.0 } else { 0.0 } };
    match op {
        BinOpKind::Add => quote! { |__a, __b| __a + __b },
        BinOpKind::Sub => quote! { |__a, __b| __a - __b },
        BinOpKind::Mul => quote! { |__a, __b| __a * __b },
        BinOpKind::Div => quote! { |__a, __b| __a / __b },
        BinOpKind::Lt => cmp(quote! { __a < __b }),
        BinOpKind::Gt => cmp(quote! { __a > __b }),
        BinOpKind::Le => cmp(quote! { __a <= __b }),
        BinOpKind::Ge => cmp(quote! { __a >= __b }),
        BinOpKind::Eq => cmp(quote! { (__a - __b).abs() < f64::EPSILON }),
        BinOpKind::Ne => cmp(quote! { (__a - __b).abs() >= f64::EPSILON }),
        BinOpKind::And => cmp(quote! { __a != 0.0 && __b != 0.0 }),
        BinOpKind::Or => cmp(quote! { __a != 0.0 || __b != 0.0 }),
    }
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
