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
    if produces_cval(ctx, expr) {
        lower_expr(ctx, expr)
    } else {
        let rt = &ctx.rt;
        let inner = lower_expr(ctx, expr)?;
        Ok(quote! { #rt::CVal::Arr(#inner) })
    }
}

/// Does this expression already lower to a `CVal` (rather than a
/// `DenseArray`)? String / record literals, the string / IO / Result
/// builtins (`ok`/`err`/`check`), and calls to user functions whose
/// body produces a `CVal` all do.
/// Builtins whose lowering yields a `CVal` (not a `DenseArray`), so
/// `lower_cval` uses their value directly instead of wrapping it in
/// `CVal::Arr`. The single source of truth for that classification --
/// adding a CVal-returning builtin is one entry here. (`tokenize_bytes`
/// is deliberately absent: it returns a `DenseArray`.)
const CVAL_BUILTINS: &[&str] = &[
    "write_stdout",
    "args",
    "arg",
    "ok",
    "err",
    "check",
    "read_bytes",
    "file_size",
    "write_bytes",
    "append_bytes",
    "decode_bytes",
    "to_int",
    "disp",
    "print",
    "eprint",
    "read_stdin",
    "type_of",
    "str_concat",
    "str_slice",
    "str_split",
];

pub(crate) fn produces_cval(ctx: &Ctx, expr: &Expr) -> bool {
    match expr {
        Expr::StrLit(..) | Expr::RecordLit { .. } => true,
        // A binding whose value produced a CVal (e.g. `b =
        // read_bytes(p)?` or `s = "text"`) carries that type forward.
        Expr::Ident(name, _) => ctx.cval_bindings.borrow().contains(name),
        Expr::FnCall { name, .. } => match name.strip_prefix("u:") {
            Some(bare) => ctx.cval_returning.contains(bare),
            None => CVAL_BUILTINS.contains(&name.as_str()),
        },
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
