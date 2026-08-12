//! FnCall lowering + static label inference (compile-to-rust step 004).
//!
//! Keeps the builtin match arms and the related helpers
//! (`extract_label_list`, `labels_of`) out of `lib.rs` so the top-
//! level module stays under the sw-checklist function-count budget.

use mlpl_parser::Expr;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::{Ctx, LowerError, lower_expr};

/// Lower a call to one of the supported builtins, or to a user
/// function (`u:name` -> the nested `user_name` fn from
/// `fndef_lower`).
pub(crate) fn lower_fncall(
    ctx: &Ctx,
    name: &str,
    args: &[Expr],
) -> Result<TokenStream, LowerError> {
    if let Some(bare) = name.strip_prefix("u:") {
        let id = format_ident!("user_{bare}");
        let lowered: Vec<TokenStream> = args
            .iter()
            .map(|a| lower_expr(ctx, a))
            .collect::<Result<_, _>>()?;
        return Ok(quote! { #id(#(#lowered),*) });
    }
    let rt = &ctx.rt;
    match (name, args.len()) {
        ("iota" | "range", 1) => {
            let arg = lower_expr(ctx, &args[0])?;
            Ok(quote! { #rt::iota((#arg).data()[0] as usize) })
        }
        ("shape", 1) => {
            let a = lower_expr(ctx, &args[0])?;
            Ok(quote! { #rt::shape(&(#a)) })
        }
        ("rank", 1) => {
            let a = lower_expr(ctx, &args[0])?;
            Ok(quote! { #rt::rank(&(#a)) })
        }
        ("transpose", 1) => {
            let a = lower_expr(ctx, &args[0])?;
            Ok(quote! { #rt::transpose(&(#a)) })
        }
        ("reshape", 2) => {
            let a = lower_expr(ctx, &args[0])?;
            let shape = lower_expr(ctx, &args[1])?;
            Ok(quote! {{
                let __shape = #shape;
                let __dims: Vec<usize> = __shape.data().iter().map(|&d| d as usize).collect();
                #rt::reshape(&(#a), &__dims).unwrap()
            }})
        }
        ("reduce_add", 1) => {
            let a = lower_expr(ctx, &args[0])?;
            Ok(quote! { #rt::reduce_add(&(#a)) })
        }
        ("reduce_add", 2) => {
            let a = lower_expr(ctx, &args[0])?;
            let axis = lower_expr(ctx, &args[1])?;
            Ok(quote! { #rt::reduce_add_axis(&(#a), (#axis).data()[0] as usize).unwrap() })
        }
        ("label" | "relabel", 2) | ("reshape_labeled", 3) => lower_label_attach(ctx, name, args),
        ("matmul", 2) => lower_matmul(ctx, args),
        ("write_stdout", 1) => {
            let a = crate::lower_cval(ctx, &args[0])?;
            Ok(quote! { #rt::write_stdout(&(#a)) })
        }
        ("ok" | "err", 1) => {
            let payload = crate::lower_cval(ctx, &args[0])?;
            let ok = name == "ok";
            Ok(quote! { #rt::CVal::result(#ok, #payload) })
        }
        ("check", 1) => lower_check(ctx, &args[0]),
        (
            "band" | "bor" | "bxor" | "bnot" | "popcount" | "shl" | "shr" | "bmask" | "bits"
            | "from_bits",
            _,
        ) => lower_bit_op(ctx, name, args),
        ("args", 0) => Ok(quote! { #rt::cli_args() }),
        ("arg", 1) => {
            let a = crate::lower_cval(ctx, &args[0])?;
            Ok(quote! { #rt::arg(&(#a)) })
        }
        _ => Err(LowerError::Unsupported(format!(
            "fncall {name}/{}",
            args.len()
        ))),
    }
}

/// Lower a bit-op call (`band`, `shl`, `from_bits`, ...) to the
/// runtime's single name/args dispatch. Arity is validated inside the
/// dispatch, so every op routes here regardless of arg count; the
/// two unwraps turn an unknown name (impossible here) and a
/// domain error into a hard panic -- matching the interpreter's
/// `RuntimeError` (bit-op domain errors are not `err` Results).
fn lower_bit_op(ctx: &Ctx, name: &str, args: &[Expr]) -> Result<TokenStream, LowerError> {
    let rt = &ctx.rt;
    let lowered: Vec<TokenStream> = args
        .iter()
        .map(|a| lower_expr(ctx, a))
        .collect::<Result<_, _>>()?;
    Ok(quote! {
        #rt::bit_try_call(#name, vec![#(#lowered),*]).expect("bit builtin").unwrap()
    })
}

/// Lower `check(expr)` -- the `?` operator. Valid only inside a
/// `CVal`-returning function: on `ok` it yields the unwrapped payload
/// (a `CVal`); on `err` it early-`return`s the whole Result from the
/// enclosing function. Outside such a function `?` has nowhere to
/// propagate, so it is `Unsupported`.
fn lower_check(ctx: &Ctx, arg: &Expr) -> Result<TokenStream, LowerError> {
    if !ctx.in_cval_fn.get() {
        return Err(LowerError::Unsupported(
            "check (the `?` operator) outside a Result-returning function \
             (define the caller with `def u:` and have it return ok/err)"
                .into(),
        ));
    }
    let rt = &ctx.rt;
    let call = lower_expr(ctx, arg)?;
    Ok(quote! {
        match #call {
            #rt::CVal::Result { ok: true, payload } => *payload,
            __err => return __err,
        }
    })
}

/// Extract a list of string literals from an `ArrayLit` -- used
/// for the label-attaching builtins. Returns `None` if the AST
/// node is not an all-StrLit ArrayLit.
pub(crate) fn extract_label_list(expr: &Expr) -> Option<Vec<String>> {
    let Expr::ArrayLit(elems, _) = expr else {
        return None;
    };
    let mut out = Vec::with_capacity(elems.len());
    for e in elems {
        let Expr::StrLit(s, _) = e else {
            return None;
        };
        out.push(s.clone());
    }
    Some(out)
}

/// Statically infer the labels of an expression where possible.
/// Returns `None` when the labels cannot be determined without
/// running the program. Used for compile-time contraction checks.
pub(crate) fn labels_of(ctx: &Ctx, expr: &Expr) -> Option<Vec<Option<String>>> {
    match expr {
        Expr::Ident(name, _) => ctx.known_labels.get(name).cloned(),
        Expr::FnCall { name, args, .. } => match (name.as_str(), args.len()) {
            ("label" | "relabel", 2) => {
                extract_label_list(&args[1]).map(|v| v.into_iter().map(Some).collect())
            }
            ("reshape_labeled", 3) => {
                extract_label_list(&args[2]).map(|v| v.into_iter().map(Some).collect())
            }
            ("transpose", 1) => labels_of(ctx, &args[0]).map(|mut v| {
                v.reverse();
                v
            }),
            _ => None,
        },
        _ => None,
    }
}

/// Lower `label`, `relabel`, and `reshape_labeled` -- the three
/// builtins that attach axis labels to an array at runtime.
fn lower_label_attach(ctx: &Ctx, name: &str, args: &[Expr]) -> Result<TokenStream, LowerError> {
    let rt = &ctx.rt;
    let labels_arg_idx = if name == "reshape_labeled" { 2 } else { 1 };
    let labels = extract_label_list(&args[labels_arg_idx])
        .ok_or_else(|| LowerError::LabelsMustBeStringLiterals(name.into()))?;
    let lits: Vec<TokenStream> = labels
        .into_iter()
        .map(|s| quote! { Some(#s.into()) })
        .collect();
    let a = lower_expr(ctx, &args[0])?;
    if name == "reshape_labeled" {
        let shape = lower_expr(ctx, &args[1])?;
        Ok(quote! {{
            let __shape = #shape;
            let __dims: Vec<usize> = __shape.data().iter().map(|&d| d as usize).collect();
            #rt::reshape(&(#a), &__dims).unwrap().with_labels(vec![#(#lits),*]).unwrap()
        }})
    } else {
        Ok(quote! { (#a).with_labels(vec![#(#lits),*]).unwrap() })
    }
}

/// Lower `matmul`, performing the static contraction-axis check
/// when both operands' labels are known at lower time. A
/// disagreement raises `LowerError::StaticShapeMismatch`; step 005
/// will map that to `compile_error!` in proc-macro context.
fn lower_matmul(ctx: &Ctx, args: &[Expr]) -> Result<TokenStream, LowerError> {
    let a_labels = labels_of(ctx, &args[0]);
    let b_labels = labels_of(ctx, &args[1]);
    if let (Some(al), Some(bl)) = (&a_labels, &b_labels)
        && al.len() == 2
        && !bl.is_empty()
        && let (Some(ac), Some(bc)) = (&al[1], &bl[0])
        && ac != bc
    {
        return Err(LowerError::StaticShapeMismatch {
            op: "matmul".into(),
            expected: al.clone(),
            actual: bl.clone(),
        });
    }
    let a = lower_expr(ctx, &args[0])?;
    let b = lower_expr(ctx, &args[1])?;
    let rt = &ctx.rt;
    Ok(quote! { #rt::MatmulExt::matmul(&(#a), &(#b)).unwrap() })
}
