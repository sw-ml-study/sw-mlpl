//! AST node types for MLPL.
//!
//! These are parser-owned syntax nodes. They do NOT depend on mlpl-array.

use mlpl_core::Span;

/// Binary operator kind.
#[derive(Clone, Debug, PartialEq)]
pub enum BinOpKind {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
}

/// An expression in the MLPL AST.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// Integer literal.
    IntLit(i64, Span),
    /// Float literal.
    FloatLit(f64, Span),
    /// String literal.
    StrLit(String, Span),
    /// Identifier reference.
    Ident(String, Span),
    /// Array literal: `[expr, expr, ...]`
    ArrayLit(Vec<Expr>, Span),
    /// Binary operation: `lhs op rhs`
    BinOp {
        /// The operator.
        op: BinOpKind,
        /// Left-hand side.
        lhs: Box<Expr>,
        /// Right-hand side.
        rhs: Box<Expr>,
        /// Span covering the full expression.
        span: Span,
    },
    /// Function call: `name(args...)`
    FnCall {
        /// Function name.
        name: String,
        /// Arguments.
        args: Vec<Expr>,
        /// Span covering name through closing paren.
        span: Span,
    },
    /// Unary negation: `-expr`
    UnaryNeg {
        /// The operand.
        operand: Box<Expr>,
        /// Span covering the minus through the operand.
        span: Span,
    },
    /// Assignment: `name = value`
    Assign {
        /// Variable name.
        name: String,
        /// Value expression.
        value: Box<Expr>,
        /// Span covering name through value.
        span: Span,
    },
    /// Repeat loop: `repeat <count> { body }`
    Repeat {
        /// Number of iterations.
        count: Box<Expr>,
        /// Body statements.
        body: Vec<Expr>,
        /// Span covering keyword through closing brace.
        span: Span,
    },
}

impl Expr {
    /// Return the source span for this expression.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::IntLit(_, s)
            | Self::FloatLit(_, s)
            | Self::StrLit(_, s)
            | Self::Ident(_, s)
            | Self::ArrayLit(_, s)
            | Self::BinOp { span: s, .. }
            | Self::UnaryNeg { span: s, .. }
            | Self::FnCall { span: s, .. }
            | Self::Assign { span: s, .. }
            | Self::Repeat { span: s, .. } => *s,
        }
    }
}
