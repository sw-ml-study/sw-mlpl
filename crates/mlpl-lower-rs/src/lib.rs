//! Lower MLPL AST to Rust `TokenStream` (compile-to-rust saga).
//!
//! Shared codegen for the future `mlpl!` proc macro (step 005) and
//! `mlpl build` subcommand (step 008). Walks an `mlpl_parser::Expr`
//! AST and emits Rust code that, when compiled, produces the same
//! numeric result as the interpreter.
//!
//! Phase-2 coverage (this step):
//! - Every MLPL expression lowers to a Rust expression of type
//!   `::mlpl_rt::DenseArray`. Scalar literals wrap via
//!   `DenseArray::from_scalar`; binary ops thread through
//!   `DenseArray::apply_binop`; unary negation goes through
//!   `DenseArray::map`. This uniform representation means later
//!   phases can swap primitives without touching the core shape.
//! - Array literals (`[1, 2, 3]`, `[[1,2],[3,4]]`) lower to
//!   `mlpl_rt::array_lit(vec![...]).unwrap()`.
//! - Variable bindings: `Assign` becomes a `let` in an outer block;
//!   `Ident` becomes `name.clone()` so the same binding can be
//!   referenced multiple times without moves.
//! - Function calls for the phase-1 `mlpl-rt` primitives: `iota`,
//!   `shape`, `rank`, `reshape`, `transpose`, `reduce_add` (flat
//!   and per-axis).
//!
//! Unsupported constructs (TensorCtor, Repeat, Train, FnCall for
//! primitives not in the phase-1 list, annotation syntax and
//! labels) return `LowerError::Unsupported` for later steps.

use mlpl_parser::{BinOpKind, Expr};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// Error produced while lowering MLPL AST to Rust.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// The AST contains a node kind not yet covered by the lowering.
    Unsupported(String),
    /// The program is empty; nothing to lower.
    EmptyProgram,
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "lower: unsupported construct: {what}"),
            Self::EmptyProgram => write!(f, "lower: empty program"),
        }
    }
}

impl std::error::Error for LowerError {}

/// Lower an MLPL AST (list of top-level statements) into a Rust
/// `TokenStream` that evaluates to a `mlpl_rt::DenseArray`.
///
/// The output is a block expression:
/// ```text
/// {
///     let x = ::mlpl_rt::iota((5.0) as usize);
///     let y = (x.clone()).apply_binop(...).unwrap();
///     y.clone()
/// }
/// ```
/// When the final statement is an `Assign`, the block yields that
/// binding's value; otherwise it yields the expression's value
/// directly, matching the interpreter's "last expression wins"
/// semantics.
pub fn lower(stmts: &[Expr]) -> Result<TokenStream, LowerError> {
    if stmts.is_empty() {
        return Err(LowerError::EmptyProgram);
    }
    let mut bindings: Vec<TokenStream> = Vec::new();
    let last_idx = stmts.len() - 1;
    let mut final_expr: Option<TokenStream> = None;
    for (i, stmt) in stmts.iter().enumerate() {
        let is_last = i == last_idx;
        if let Expr::Assign { name, value, .. } = stmt {
            let val = lower_expr(value)?;
            let id = format_ident!("{name}");
            bindings.push(quote! { let #id = #val; });
            if is_last {
                final_expr = Some(quote! { #id.clone() });
            }
        } else {
            let val = lower_expr(stmt)?;
            if is_last {
                final_expr = Some(val);
            } else {
                bindings.push(quote! { let _ = #val; });
            }
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

/// Lower a single expression to a Rust expression of type
/// `::mlpl_rt::DenseArray`.
fn lower_expr(expr: &Expr) -> Result<TokenStream, LowerError> {
    match expr {
        Expr::IntLit(n, _) => {
            let v = *n as f64;
            Ok(quote! { ::mlpl_rt::DenseArray::from_scalar(#v) })
        }
        Expr::FloatLit(f, _) => {
            let v = *f;
            Ok(quote! { ::mlpl_rt::DenseArray::from_scalar(#v) })
        }
        Expr::UnaryNeg { operand, .. } => {
            let inner = lower_expr(operand)?;
            Ok(quote! { (#inner).map(|__v| -__v) })
        }
        Expr::BinOp { op, lhs, rhs, .. } => {
            let l = lower_expr(lhs)?;
            let r = lower_expr(rhs)?;
            let closure = match op {
                BinOpKind::Add => quote! { |__a, __b| __a + __b },
                BinOpKind::Sub => quote! { |__a, __b| __a - __b },
                BinOpKind::Mul => quote! { |__a, __b| __a * __b },
                BinOpKind::Div => quote! { |__a, __b| __a / __b },
            };
            Ok(quote! { (#l).apply_binop(&(#r), #closure).unwrap() })
        }
        Expr::Ident(name, _) => {
            let id = format_ident!("{name}");
            Ok(quote! { #id.clone() })
        }
        Expr::ArrayLit(elems, _) => {
            let lowered: Vec<TokenStream> =
                elems.iter().map(lower_expr).collect::<Result<_, _>>()?;
            Ok(quote! { ::mlpl_rt::array_lit(vec![#(#lowered),*]).unwrap() })
        }
        Expr::FnCall { name, args, .. } => lower_fncall(name, args),
        Expr::Assign { .. } => Err(LowerError::Unsupported(
            "nested assignment (assignment as subexpression)".into(),
        )),
        Expr::StrLit(_, _) | Expr::TensorCtor { .. } | Expr::Repeat { .. } | Expr::Train { .. } => {
            Err(LowerError::Unsupported(format!("{expr:?}")))
        }
    }
}

/// Lower a call to one of the phase-1 `mlpl-rt` primitives.
fn lower_fncall(name: &str, args: &[Expr]) -> Result<TokenStream, LowerError> {
    match (name, args.len()) {
        ("iota", 1) => {
            let arg = lower_expr(&args[0])?;
            Ok(quote! { ::mlpl_rt::iota((#arg).data()[0] as usize) })
        }
        ("shape", 1) => {
            let a = lower_expr(&args[0])?;
            Ok(quote! { ::mlpl_rt::shape(&(#a)) })
        }
        ("rank", 1) => {
            let a = lower_expr(&args[0])?;
            Ok(quote! { ::mlpl_rt::rank(&(#a)) })
        }
        ("transpose", 1) => {
            let a = lower_expr(&args[0])?;
            Ok(quote! { ::mlpl_rt::transpose(&(#a)) })
        }
        ("reshape", 2) => {
            let a = lower_expr(&args[0])?;
            let shape = lower_expr(&args[1])?;
            Ok(quote! {{
                let __shape = #shape;
                let __dims: Vec<usize> = __shape.data().iter().map(|&d| d as usize).collect();
                ::mlpl_rt::reshape(&(#a), &__dims).unwrap()
            }})
        }
        ("reduce_add", 1) => {
            let a = lower_expr(&args[0])?;
            Ok(quote! { ::mlpl_rt::reduce_add(&(#a)) })
        }
        ("reduce_add", 2) => {
            let a = lower_expr(&args[0])?;
            let axis = lower_expr(&args[1])?;
            Ok(quote! { ::mlpl_rt::reduce_add_axis(&(#a), (#axis).data()[0] as usize).unwrap() })
        }
        _ => Err(LowerError::Unsupported(format!(
            "fncall {name}/{}",
            args.len()
        ))),
    }
}
