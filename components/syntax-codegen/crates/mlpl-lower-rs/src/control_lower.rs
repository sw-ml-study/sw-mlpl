//! Lowering control-flow expressions (compiler-control-flow). Branch
//! (and, later, loop) bodies reuse `fndef_lower::lower_body`, so
//! `return` inside a branch emits a real Rust return. Truthiness
//! follows the interpreter: a non-zero scalar is true.

use mlpl_parser::Expr;
use proc_macro2::TokenStream;
use quote::quote;

use crate::{Ctx, LowerError, fndef_lower, lower_expr};

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
    let t = fndef_lower::lower_body(ctx, then_body)?;
    let e = fndef_lower::lower_body(ctx, else_body)?;
    Ok(quote! { if (#c).data()[0] != 0.0 { #t } else { #e } })
}
