//! Program + statement + expression lowering: the walk that emits
//! Rust `TokenStream` from an MLPL AST. Data types live in `model`,
//! control flow in `control_lower`, user functions in `fndef_lower`,
//! builtin calls in `fncall`; lib.rs is a facade.

use mlpl_parser::{BinOpKind, Expr};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::{
    Ctx, LowerConfig, LowerError, control_lower, fndef_lower, labels_of, lower_cval, lower_field,
    lower_fncall, lower_record,
};

/// Lower an MLPL AST using the default configuration
/// (`::mlpl_rt::...` paths). Shorthand for
/// `lower_with_config(stmts, &LowerConfig::default())`.
pub fn lower(stmts: &[Expr]) -> Result<TokenStream, LowerError> {
    lower_with_config(stmts, &LowerConfig::default())
}

/// Lower an MLPL AST with a configurable runtime path. Used by the
/// `mlpl!` proc macro (step 005) to emit paths through a facade
/// crate instead of the raw `mlpl-rt` crate.
pub fn lower_with_config(stmts: &[Expr], cfg: &LowerConfig) -> Result<TokenStream, LowerError> {
    if stmts.is_empty() {
        return Err(LowerError::EmptyProgram);
    }
    let mut ctx = Ctx::new(cfg);
    ctx.cval_returning = control_lower::collect_cval_returning(stmts);
    let mut bindings: Vec<TokenStream> = Vec::new();
    let last_idx = stmts.len() - 1;
    let mut final_expr: Option<TokenStream> = None;
    for (i, stmt) in stmts.iter().enumerate() {
        let is_last = i == last_idx;
        if let Some(fin) = lower_stmt(&mut ctx, stmt, is_last, &mut bindings)? {
            final_expr = Some(fin);
        }
    }
    let final_value = final_expr.expect("final expr populated");
    Ok(quote! {
        {
            #(#bindings)*
            #final_value
        }
    })
}

/// Lower one statement into `bindings`, returning `Some(tokens)`
/// when it is the final statement (whose value the block yields).
fn lower_stmt(
    ctx: &mut Ctx,
    stmt: &Expr,
    is_last: bool,
    bindings: &mut Vec<TokenStream>,
) -> Result<Option<TokenStream>, LowerError> {
    if let Expr::Assign { name, value, .. } = stmt {
        return lower_assign(ctx, name, value, is_last, bindings);
    }
    if let Expr::FnDef {
        name, params, body, ..
    } = stmt
    {
        // A `def u:` lowers to a nested Rust fn item (hoisted, so
        // call order is free) and yields no program value.
        bindings.push(fndef_lower::lower_user_fn(ctx, name, params, body)?);
        return Ok(None);
    }
    if is_last {
        return Ok(Some(lower_cval(ctx, stmt)?));
    }
    let val = lower_expr(ctx, stmt)?;
    bindings.push(quote! { let _ = #val; });
    Ok(None)
}

/// Lower `name = value` into a `let` binding (recording any static
/// labels), yielding `CVal::Arr(name)` when it is the program's
/// final statement.
fn lower_assign(
    ctx: &mut Ctx,
    name: &str,
    value: &Expr,
    is_last: bool,
    bindings: &mut Vec<TokenStream>,
) -> Result<Option<TokenStream>, LowerError> {
    let val = lower_expr(ctx, value)?;
    if let Some(lbls) = labels_of(ctx, value) {
        ctx.known_labels.insert(name.to_string(), lbls);
    }
    let id = format_ident!("{name}");
    // First assignment in scope -> `let mut` (so a later rebind can
    // reassign, e.g. a loop accumulator); a rebind -> reassignment.
    bindings.push(if ctx.first_binding(name) {
        quote! { let mut #id = #val; }
    } else {
        quote! { #id = #val; }
    });
    let rt = &ctx.rt;
    Ok(is_last.then(|| quote! { #rt::CVal::Arr(#id.clone()) }))
}

/// Lower a single expression. Returns a `DenseArray`-valued Rust
/// expression. Fails on unsupported constructs or static label
/// mismatches.
pub(crate) fn lower_expr(ctx: &Ctx, expr: &Expr) -> Result<TokenStream, LowerError> {
    match expr {
        Expr::Include(path, _) => Err(LowerError::Unsupported(format!(
            "include \"{path}\" (the compile path takes one already-resolved source; \
             run the loader first or hand-splice)"
        ))),
        Expr::IntLit(n, _) => {
            let rt = &ctx.rt;
            let v = *n as f64;
            Ok(quote! { #rt::DenseArray::from_scalar(#v) })
        }
        Expr::FloatLit(f, _) => {
            let rt = &ctx.rt;
            let v = *f;
            Ok(quote! { #rt::DenseArray::from_scalar(#v) })
        }
        Expr::UnaryNeg { operand, .. } => {
            let inner = lower_expr(ctx, operand)?;
            Ok(quote! { (#inner).map(|__v| -__v) })
        }
        Expr::BinOp { op, lhs, rhs, .. } => {
            let l = lower_expr(ctx, lhs)?;
            let r = lower_expr(ctx, rhs)?;
            let closure = match op {
                BinOpKind::Add => quote! { |__a, __b| __a + __b },
                BinOpKind::Sub => quote! { |__a, __b| __a - __b },
                BinOpKind::Mul => quote! { |__a, __b| __a * __b },
                BinOpKind::Div => quote! { |__a, __b| __a / __b },
            };
            let rt = &ctx.rt;
            // UFCS through the runtime facade's re-exported trait, so
            // the generated call site needs no `use ApplyBinopExt`.
            Ok(quote! { #rt::ApplyBinopExt::apply_binop(&(#l), &(#r), #closure).unwrap() })
        }
        Expr::Ident(name, _) => {
            let id = format_ident!("{name}");
            Ok(quote! { #id.clone() })
        }
        Expr::ArrayLit(elems, _) => {
            let rt = &ctx.rt;
            let lowered: Vec<TokenStream> = elems
                .iter()
                .map(|e| lower_expr(ctx, e))
                .collect::<Result<_, _>>()?;
            Ok(quote! { #rt::array_lit(vec![#(#lowered),*]).unwrap() })
        }
        Expr::FnCall { name, args, .. } => lower_fncall(ctx, name, args),
        Expr::Assign { .. } => Err(LowerError::Unsupported(
            "nested assignment (assignment as subexpression)".into(),
        )),
        Expr::StrLit(s, _) => {
            let rt = &ctx.rt;
            Ok(quote! { #rt::CVal::Str(#s.to_string()) })
        }
        Expr::If {
            cond,
            then_body,
            else_body,
            ..
        } => control_lower::lower_if(ctx, cond, then_body, else_body),
        Expr::While { cond, body, .. } => control_lower::lower_while(ctx, cond, body),
        Expr::RecordLit { fields, .. } => lower_record(ctx, fields),
        Expr::FieldAccess {
            receiver, field, ..
        } => lower_field(ctx, receiver, field),
        Expr::BuiltinRef(_, _)
        | Expr::TensorCtor { .. }
        | Expr::Repeat { .. }
        | Expr::Train { .. }
        | Expr::For { .. }
        | Expr::Experiment { .. }
        | Expr::Device { .. }
        | Expr::Break { .. }
        | Expr::Continue { .. }
        | Expr::FnDef { .. }
        | Expr::TryCatch { .. }
        | Expr::Return { .. } => Err(LowerError::Unsupported(format!("{expr:?}"))),
    }
}
