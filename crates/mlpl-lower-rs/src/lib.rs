//! Lower MLPL AST to Rust `TokenStream` (compile-to-rust saga).
//!
//! Shared codegen for the future `mlpl!` proc macro (step 005) and
//! `mlpl build` subcommand (step 008). Walks an `mlpl_parser::Expr`
//! AST and emits Rust code that, when compiled, produces the same
//! numeric result as the interpreter.
//!
//! Phase-2 coverage (through step 004):
//! - Every MLPL expression lowers to a Rust expression of type
//!   `::mlpl_rt::DenseArray`. Scalar literals wrap via
//!   `DenseArray::from_scalar`; binary ops thread through
//!   `DenseArray::apply_binop`; unary negation goes through
//!   `DenseArray::map`.
//! - Array literals lower to `mlpl_rt::array_lit(...)`.
//! - Variable bindings: `Assign` becomes a `let`; `Ident` becomes
//!   `name.clone()`.
//! - FnCalls: `iota`, `shape`, `rank`, `reshape`, `transpose`,
//!   `reduce_add` (flat + axis), plus the label builtins `label`,
//!   `relabel`, `reshape_labeled`, and `matmul`.
//! - Annotation syntax `x : [batch, dim] = ...` is automatic
//!   because the parser desugars it to `label(<value>, [...])`.
//! - **Static label check for matmul**: when both operands' labels
//!   are known at lower time, the contraction axis is checked and
//!   a mismatch surfaces as `LowerError::StaticShapeMismatch`. The
//!   proc macro (step 005) will convert this to `compile_error!`
//!   with span-preserved carats.
//!
//! Known deferred constructs (`labels()` as a REPL-only string
//! builtin, `TensorCtor`, `Repeat`, `Train`) still return
//! `LowerError::Unsupported`.

use std::collections::HashMap;

use mlpl_parser::{BinOpKind, Expr};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

mod fncall;

use fncall::{labels_of, lower_fncall};

/// Error produced while lowering MLPL AST to Rust.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// The AST contains a node kind not yet covered by the lowering.
    Unsupported(String),
    /// The program is empty; nothing to lower.
    EmptyProgram,
    /// Two operands' labels statically disagree on an operator
    /// that requires them to match (currently: `matmul` contraction
    /// axis). Surfaces at lower time; step 005 maps this to
    /// `compile_error!` in a proc-macro context.
    StaticShapeMismatch {
        /// Operator or builtin name.
        op: String,
        /// Labels on the first operand at lower time.
        expected: Vec<Option<String>>,
        /// Labels on the second operand at lower time.
        actual: Vec<Option<String>>,
    },
    /// A label-attaching builtin (`label`, `relabel`,
    /// `reshape_labeled`) was called with a label list that is not
    /// a bracketed list of string literals.
    LabelsMustBeStringLiterals(String),
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "lower: unsupported construct: {what}"),
            Self::EmptyProgram => write!(f, "lower: empty program"),
            Self::StaticShapeMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "lower: {op} static label mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::LabelsMustBeStringLiterals(fn_name) => write!(
                f,
                "lower: {fn_name}: label list must be [\"name1\", \"name2\", ...]"
            ),
        }
    }
}

impl std::error::Error for LowerError {}

/// Compile-time knowledge of a binding's axis labels, built up as
/// we walk top-level statements. When a name is missing the
/// static check is skipped; the runtime still validates. Saga:
/// compile-to-rust step 004.
/// Configuration for lowering. Default emits `::mlpl_rt::...` paths
/// (direct runtime dep); proc-macro users override `rt_path` to a
/// re-exported path through their facade crate.
pub struct LowerConfig {
    /// Path token sequence to the runtime, e.g. `::mlpl_rt` or
    /// `::mlpl::__rt`. Prefixed before every primitive call.
    pub rt_path: TokenStream,
}

impl Default for LowerConfig {
    fn default() -> Self {
        Self {
            rt_path: quote! { ::mlpl_rt },
        }
    }
}

pub(crate) struct Ctx {
    pub(crate) known_labels: HashMap<String, Vec<Option<String>>>,
    pub(crate) rt: TokenStream,
}

impl Ctx {
    fn new(cfg: &LowerConfig) -> Self {
        Self {
            known_labels: HashMap::new(),
            rt: cfg.rt_path.clone(),
        }
    }
}

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
    let mut bindings: Vec<TokenStream> = Vec::new();
    let last_idx = stmts.len() - 1;
    let mut final_expr: Option<TokenStream> = None;
    for (i, stmt) in stmts.iter().enumerate() {
        let is_last = i == last_idx;
        if let Expr::Assign { name, value, .. } = stmt {
            let val = lower_expr(&ctx, value)?;
            if let Some(lbls) = labels_of(&ctx, value) {
                ctx.known_labels.insert(name.clone(), lbls);
            }
            let id = format_ident!("{name}");
            bindings.push(quote! { let #id = #val; });
            if is_last {
                final_expr = Some(quote! { #id.clone() });
            }
        } else {
            let val = lower_expr(&ctx, stmt)?;
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

/// Lower a single expression. Returns a `DenseArray`-valued Rust
/// expression. Fails on unsupported constructs or static label
/// mismatches.
pub(crate) fn lower_expr(ctx: &Ctx, expr: &Expr) -> Result<TokenStream, LowerError> {
    match expr {
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
            Ok(quote! { (#l).apply_binop(&(#r), #closure).unwrap() })
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
        Expr::StrLit(_, _)
        | Expr::TensorCtor { .. }
        | Expr::Repeat { .. }
        | Expr::Train { .. }
        | Expr::For { .. }
        | Expr::Experiment { .. }
        | Expr::Device { .. } => Err(LowerError::Unsupported(format!("{expr:?}"))),
    }
}
