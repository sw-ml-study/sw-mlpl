//! Lower MLPL AST to Rust `TokenStream` (compile-to-rust saga).
//!
//! This is the codegen crate shared by the future `mlpl!` proc
//! macro and the `mlpl build` subcommand. It walks a `mlpl_parser`
//! AST and emits Rust code that, when compiled, produces the same
//! numeric result as the interpreter.
//!
//! Phase 1 (this step) covers the scalar-arithmetic subset only:
//! `IntLit`, `FloatLit`, `UnaryNeg`, and `BinOp(+, -, *, /)`. The
//! emitted code uses native `f64` arithmetic so the compiler can
//! inline and constant-fold freely; the outermost scalar is wrapped
//! into `mlpl_rt::DenseArray::from_scalar(...)` so the macro's
//! return type stays uniform with the forthcoming array phases.
//!
//! Unsupported constructs (Ident, Assign, FnCall, ArrayLit, ...)
//! return `LowerError::Unsupported` for now. Later saga steps
//! extend the match arms.

use mlpl_parser::{BinOpKind, Expr};
use proc_macro2::TokenStream;
use quote::quote;

/// Error produced while lowering MLPL AST to Rust.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// The AST contains a node kind not yet covered by the lowering.
    Unsupported(String),
    /// The program is empty; nothing to lower.
    EmptyProgram,
    /// This phase only supports single-expression programs; multi-stmt
    /// lands in step 003 alongside variable bindings.
    MultiStatementNotYetSupported,
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "lower: unsupported construct: {what}"),
            Self::EmptyProgram => write!(f, "lower: empty program"),
            Self::MultiStatementNotYetSupported => write!(
                f,
                "lower: multi-statement programs not yet supported (phase 1)"
            ),
        }
    }
}

impl std::error::Error for LowerError {}

/// Lower an MLPL AST (list of top-level statements) into a Rust
/// `TokenStream` that evaluates to a `mlpl_rt::DenseArray`.
///
/// Phase 1 accepts exactly one statement, which must be a scalar
/// expression. The returned tokens, when placed in expression
/// position inside a function that has `use mlpl_rt;` in scope,
/// evaluate to a `DenseArray` scalar with the expected value.
pub fn lower(stmts: &[Expr]) -> Result<TokenStream, LowerError> {
    if stmts.is_empty() {
        return Err(LowerError::EmptyProgram);
    }
    if stmts.len() > 1 {
        return Err(LowerError::MultiStatementNotYetSupported);
    }
    let scalar = lower_scalar(&stmts[0])?;
    Ok(quote! {
        ::mlpl_rt::DenseArray::from_scalar(#scalar)
    })
}

/// Lower a scalar expression to a Rust expression of type `f64`.
fn lower_scalar(expr: &Expr) -> Result<TokenStream, LowerError> {
    match expr {
        Expr::IntLit(n, _) => {
            let v = *n as f64;
            Ok(quote! { (#v) })
        }
        Expr::FloatLit(f, _) => {
            let v = *f;
            Ok(quote! { (#v) })
        }
        Expr::UnaryNeg { operand, .. } => {
            let inner = lower_scalar(operand)?;
            Ok(quote! { (-#inner) })
        }
        Expr::BinOp { op, lhs, rhs, .. } => {
            let l = lower_scalar(lhs)?;
            let r = lower_scalar(rhs)?;
            Ok(match op {
                BinOpKind::Add => quote! { (#l + #r) },
                BinOpKind::Sub => quote! { (#l - #r) },
                BinOpKind::Mul => quote! { (#l * #r) },
                BinOpKind::Div => quote! { (#l / #r) },
            })
        }
        other => Err(LowerError::Unsupported(format!("{other:?}"))),
    }
}
