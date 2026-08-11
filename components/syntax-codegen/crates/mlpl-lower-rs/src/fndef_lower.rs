//! Lowering `def u:` user functions (compiler-functions, param-only
//! slice). A user function lowers to a nested Rust `fn` over its
//! parameters; a call `u:name(args)` routes to `user_name(args)`
//! (see `fncall`). Rust hoists nested fn items, so call order is
//! free. Reading a free/global variable is a clear `Unsupported`
//! error -- this slice supports parameters + body-local bindings
//! only (globals, control-flow-in-body, and records/results are
//! later rungs). The DenseArray value model applies.

use std::collections::HashSet;

use mlpl_parser::Expr;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::{Ctx, LowerError, lower_expr};

/// Lower `def u:name(params) { body }` into a nested Rust fn item.
pub(crate) fn lower_user_fn(
    ctx: &Ctx,
    name: &str,
    params: &[String],
    body: &[Expr],
) -> Result<TokenStream, LowerError> {
    check_no_free_vars(params, body)?;
    let fn_id = format_ident!("user_{}", name.strip_prefix("u:").unwrap_or(name));
    let param_ids: Vec<_> = params.iter().map(|p| format_ident!("{p}")).collect();
    let rt = &ctx.rt;
    // A fresh declared-scope pre-seeded with the params (so body
    // locals never alias the enclosing program's variables). Params
    // are `mut` so a body may rebind one (harmless unused_mut in the
    // generated code otherwise).
    let block = ctx.with_scope(params, || lower_body(ctx, body))?;
    Ok(quote! {
        fn #fn_id(#(mut #param_ids: #rt::DenseArray),*) -> #rt::DenseArray #block
    })
}

/// Lower a statement block (a function body or an `if` branch) into
/// a `{ ...; tail }` block whose tail expression is its DenseArray
/// value; `return` statements emit real Rust returns.
pub(crate) fn lower_body(ctx: &Ctx, body: &[Expr]) -> Result<TokenStream, LowerError> {
    let mut binds: Vec<TokenStream> = Vec::new();
    let last = body.len().saturating_sub(1);
    let mut tail: Option<TokenStream> = None;
    for (i, stmt) in body.iter().enumerate() {
        match stmt {
            Expr::Assign { name, value, .. } => {
                let (id, val) = (format_ident!("{name}"), lower_expr(ctx, value)?);
                binds.push(if ctx.first_binding(name) {
                    quote! { let mut #id = #val; }
                } else {
                    quote! { #id = #val; }
                });
            }
            // Real Rust `return` so an early return inside a branch
            // exits the enclosing fn (a diverging branch unifies with
            // the other branch's type). Bare `return` -> lower_expr err.
            Expr::Return { value: Some(v), .. } => {
                let ts = lower_expr(ctx, v)?;
                binds.push(quote! { return #ts; });
            }
            _ if i == last => tail = Some(lower_expr(ctx, stmt)?),
            _ => {
                let v = lower_expr(ctx, stmt)?;
                binds.push(quote! { let _ = #v; });
            }
        }
    }
    Ok(quote! { { #(#binds)* #tail } })
}

/// Reject a body that reads any name that is not a parameter or a
/// body-local binding (a global read), so a compiled function can
/// never silently diverge from the interpreter's snapshot scope.
fn check_no_free_vars(params: &[String], body: &[Expr]) -> Result<(), LowerError> {
    let mut bound: HashSet<&str> = params.iter().map(String::as_str).collect();
    for stmt in body {
        if let Expr::Assign { name, .. } = stmt {
            bound.insert(name);
        }
    }
    let mut free: Vec<String> = Vec::new();
    for stmt in body {
        collect_free(stmt, &bound, &mut free);
    }
    match free.first() {
        Some(v) => Err(LowerError::Unsupported(format!(
            "compiled user function reads '{v}', which is not a parameter or a local -- \
             reading a global from a compiled function is not supported yet; pass it as a parameter"
        ))),
        None => Ok(()),
    }
}

/// Collect identifier reads not in `bound` from the param-only
/// supported expression set (other node kinds are rejected by
/// `lower_expr`, so they need no walk here).
fn collect_free(expr: &Expr, bound: &HashSet<&str>, out: &mut Vec<String>) {
    match expr {
        Expr::Ident(name, _) if !bound.contains(name.as_str()) => out.push(name.clone()),
        Expr::UnaryNeg { operand, .. } => collect_free(operand, bound, out),
        Expr::BinOp { lhs, rhs, .. } => {
            collect_free(lhs, bound, out);
            collect_free(rhs, bound, out);
        }
        Expr::FnCall { args, .. } => args.iter().for_each(|a| collect_free(a, bound, out)),
        Expr::ArrayLit(elems, _) => elems.iter().for_each(|e| collect_free(e, bound, out)),
        Expr::Assign { value, .. } => collect_free(value, bound, out),
        Expr::Return { value: Some(v), .. } => collect_free(v, bound, out),
        _ => {}
    }
}
