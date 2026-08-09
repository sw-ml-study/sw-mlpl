//! Lowering into the compiled value model (`CVal`). Kept out of
//! `lib.rs` so the top-level module stays within the function-count
//! budget.

use mlpl_parser::Expr;
use proc_macro2::TokenStream;
use quote::quote;

use crate::{Ctx, LowerError, lower_expr};

/// Lower an expression in a CVal position (a `write_stdout`
/// argument or the program result): string / IO expressions
/// already produce a `CVal`; a numeric expression is wrapped as
/// `CVal::Arr`.
pub(crate) fn lower_cval(ctx: &Ctx, expr: &Expr) -> Result<TokenStream, LowerError> {
    if produces_cval(expr) {
        lower_expr(ctx, expr)
    } else {
        let rt = &ctx.rt;
        let inner = lower_expr(ctx, expr)?;
        Ok(quote! { #rt::CVal::Arr(#inner) })
    }
}

/// Does this expression already lower to a `CVal` (rather than a
/// `DenseArray`)? String literals and the string/IO builtins do.
fn produces_cval(expr: &Expr) -> bool {
    match expr {
        Expr::StrLit(..) => true,
        Expr::FnCall { name, .. } => matches!(name.as_str(), "write_stdout" | "args" | "arg"),
        _ => false,
    }
}
