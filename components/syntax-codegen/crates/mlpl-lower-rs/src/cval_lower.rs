//! Lowering into the compiled value model (`CVal`): the `CVal::Arr`
//! wrapping at value boundaries, plus record construction / field
//! projection (records are a `CVal` variant). Kept out of `lib.rs`
//! so the top-level module stays within the function-count budget.

use mlpl_parser::Expr;
use proc_macro2::TokenStream;
use quote::quote;

use crate::{Ctx, LowerError, lower_expr};

/// Lower an expression in a CVal position (a `write_stdout`
/// argument or the program result): string / IO / record / Result
/// expressions already produce a `CVal`; a numeric expression is
/// wrapped as `CVal::Arr`.
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
/// `DenseArray`)? String literals, record literals, and the string /
/// IO / Result builtins do.
fn produces_cval(expr: &Expr) -> bool {
    match expr {
        Expr::StrLit(..) | Expr::RecordLit { .. } => true,
        Expr::FnCall { name, .. } => {
            matches!(
                name.as_str(),
                "write_stdout" | "args" | "arg" | "ok" | "err"
            )
        }
        _ => false,
    }
}

/// Lower `{name: value, ...}` to `CVal::record(vec![(name, cval), ...])`.
/// Each field value lowers in CVal position (numeric fields wrap as
/// `CVal::Arr`, strings stay `CVal::Str`, nested records recurse).
pub(crate) fn lower_record(
    ctx: &Ctx,
    fields: &[(String, Expr)],
) -> Result<TokenStream, LowerError> {
    let rt = &ctx.rt;
    let entries: Vec<TokenStream> = fields
        .iter()
        .map(|(name, val)| {
            let v = lower_cval(ctx, val)?;
            Ok(quote! { (#name.to_string(), #v) })
        })
        .collect::<Result<_, LowerError>>()?;
    Ok(quote! { #rt::CVal::record(vec![#(#entries),*]) })
}

/// Lower `receiver.field` -- select a record field and unwrap its
/// numeric payload. The receiver lowers to a record `CVal`;
/// `.field(name)` selects the field and `.arr().clone()` yields its
/// `DenseArray` leaf (records-of-records / string leaves are a later
/// rung; this path assumes a numeric field).
pub(crate) fn lower_field(
    ctx: &Ctx,
    receiver: &Expr,
    field: &str,
) -> Result<TokenStream, LowerError> {
    let recv = lower_expr(ctx, receiver)?;
    Ok(quote! { (#recv).field(#field).arr().clone() })
}
