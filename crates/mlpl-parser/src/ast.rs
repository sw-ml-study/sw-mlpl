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

/// Kind of tensor constructor: trainable parameter or non-trainable tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorCtorKind {
    /// `param[shape]` -- trainable leaf (`requires_grad` = true).
    Param,
    /// `tensor[shape]` -- non-trainable leaf.
    Tensor,
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
    /// Tensor constructor: `param[shape...]` or `tensor[shape...]`.
    TensorCtor {
        /// Which kind of leaf to construct.
        kind: TensorCtorKind,
        /// Shape dimension expressions.
        shape: Vec<Expr>,
        /// Span covering keyword through closing bracket.
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
    /// Training loop: `train <count> { body }`. On each iteration the
    /// loop binds the iteration index to `step`, runs the body, and
    /// captures the value of the body's final statement as the
    /// per-step loss. After the loop, all captured losses are stored
    /// in the environment as a 1-D array under the name `last_losses`.
    Train {
        /// Number of training steps.
        count: Box<Expr>,
        /// Body statements; the value of the last one is the loss.
        body: Vec<Expr>,
        /// Span covering keyword through closing brace.
        span: Span,
    },
    /// Scoped experiment block: `experiment "name" { body }`
    /// (Saga 12 step 007). Runs body in the current environment;
    /// on exit, scans `_metric`-suffixed scalar vars and appends a
    /// record to `env.experiment_log`. When the environment has an
    /// `exp_dir` set (terminal REPL only), also writes a
    /// `run.json` record to disk.
    Experiment {
        /// Human-chosen name for the run; used in file paths.
        name: String,
        /// Body statements.
        body: Vec<Expr>,
        /// Span covering the keyword through the closing brace.
        span: Span,
    },
    /// Streaming iteration: `for <binding> in <source> { body }`
    /// (Saga 12 step 003). On each iteration binds `binding` to a
    /// rank-(r-1) slice of `source`'s axis 0. After the loop, each
    /// iteration's final value is captured into `last_rows` in the
    /// environment (mirrors `Train`'s `last_losses`).
    For {
        /// Name to bind to each row slice.
        binding: String,
        /// Source expression (must have rank >= 1).
        source: Box<Expr>,
        /// Body statements.
        body: Vec<Expr>,
        /// Span covering keyword through closing brace.
        span: Span,
    },
    /// Scoped device block: `device("mlx") { body }` or
    /// `device("cpu") { body }` (Saga 14 step 004). Inside the
    /// body, the evaluator dispatches array ops through the named
    /// runtime target -- `mlpl-mlx` when the `mlx` feature is
    /// compiled in and the block's target is `"mlx"`, else the
    /// CPU path (with a one-time warning if the user asked for MLX
    /// but the feature is unavailable). `device("cpu") { ... }` is
    /// always a no-op and works on every host.
    Device {
        /// Runtime target name (`"mlx"` or `"cpu"`).
        target: String,
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
            | Self::TensorCtor { span: s, .. }
            | Self::Repeat { span: s, .. }
            | Self::Train { span: s, .. }
            | Self::For { span: s, .. }
            | Self::Experiment { span: s, .. }
            | Self::Device { span: s, .. } => *s,
        }
    }
}
