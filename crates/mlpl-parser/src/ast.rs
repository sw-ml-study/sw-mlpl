//! AST node types for MLPL.
//!
//! These are parser-owned syntax nodes. They do NOT depend on mlpl-array.
//! `Display` impls and rendering helpers live in `ast_fmt.rs`.

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
    /// Builtin / operator reference: `:foo`, `:+`, `:max`.
    /// Evaluates to a `Value::BuiltinRef { name }` -- the
    /// canonical first-class-ish reference to a builtin or
    /// operator. Higher-order builtins like `reduce` and
    /// `map` accept this in place of (the future) function
    /// values.
    BuiltinRef(String, Span),
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
    /// runtime target -- `mlpl-mlx-rt` when the `mlx` feature is
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
    /// Record literal: `{ field1: expr1, field2: expr2, ... }`.
    /// Saga 29 step 001. Distinct from `{ stmt; ... }` blocks
    /// (which only appear after the `repeat` / `train` / `for` /
    /// `experiment` / `device` keywords); in expression position
    /// `{` always opens a record. Field names are idents.
    RecordLit {
        /// Field name / value pairs, in source order. Duplicate
        /// names error at parse time.
        fields: Vec<(String, Expr)>,
        /// Span covering opening through closing brace.
        span: Span,
    },
    /// Field access: `receiver.field`. Saga 29 step 001. Lower
    /// precedence than function call so `f(x).y` works.
    FieldAccess {
        /// Expression whose field is being read.
        receiver: Box<Expr>,
        /// Field name (always an ident).
        field: String,
        /// Span covering receiver-start through field-end.
        span: Span,
    },
    /// `if cond { then } else { else_ }` expression. Saga 31
    /// step 004. Returns the value of whichever branch was
    /// taken; both `then` and `else_` are body sequences (the
    /// final expression's value is the branch value, matching
    /// `repeat` / `train` body semantics). `else` is required.
    If {
        /// Condition expression. Truthy iff non-zero scalar or
        /// `Ok(_)` Result; everything else is an eval error.
        cond: Box<Expr>,
        /// `then` body.
        then_body: Vec<Expr>,
        /// `else` body.
        else_body: Vec<Expr>,
        /// Span covering `if` keyword through closing `else { }`.
        span: Span,
    },
    /// `while cond { body }` loop. Saga 31 step 005. Body
    /// re-evaluated until `cond` is falsy (zero scalar or `Err`)
    /// or a `break` is hit. The whole expression evaluates to
    /// the break value (default `0`) or `0` if the loop exited
    /// normally. `cond` truthiness uses the same rule as `if`.
    While {
        /// Loop condition, re-evaluated each iteration.
        cond: Box<Expr>,
        /// Loop body; statements separated by `;` or newline.
        body: Vec<Expr>,
        /// Span covering `while` keyword through closing `}`.
        span: Span,
    },
    /// `break` or `break value` -- exits the nearest enclosing
    /// `while` loop. The optional value becomes the value of the
    /// `while` expression; with no value the loop yields `0`.
    Break {
        /// Optional break value; `None` is equivalent to scalar `0`.
        value: Option<Box<Expr>>,
        /// Span of the `break` keyword and (if present) its value.
        span: Span,
    },
    /// `continue` -- skips the rest of the current `while`
    /// body and re-checks the condition.
    Continue {
        /// Span of the `continue` keyword.
        span: Span,
    },
    /// `def ns:name(param1, param2) { body }` -- user-defined
    /// function. Name must contain a colon (namespace prefix).
    FnDef {
        /// Full function name including namespace (e.g. `u:area`).
        name: String,
        /// Parameter names.
        params: Vec<String>,
        /// Body expressions; last value is the return value.
        body: Vec<Expr>,
        /// Span covering `def` through closing `}`.
        span: Span,
    },
    /// `return expr` -- early exit from a UDF body. Without a
    /// value, returns scalar 0 (same as bare `break`).
    Return {
        /// Optional return value.
        value: Option<Box<Expr>>,
        /// Span of `return` keyword and optional value.
        span: Span,
    },
}

impl Expr {
    /// Return the source span for this expression.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::IntLit(_, s) | Self::FloatLit(_, s) | Self::StrLit(_, s) => *s,
            Self::Ident(_, s) | Self::BuiltinRef(_, s) | Self::ArrayLit(_, s) => *s,
            Self::BinOp { span, .. }
            | Self::UnaryNeg { span, .. }
            | Self::FnCall { span, .. }
            | Self::Assign { span, .. }
            | Self::TensorCtor { span, .. }
            | Self::Repeat { span, .. }
            | Self::Train { span, .. }
            | Self::For { span, .. }
            | Self::Experiment { span, .. }
            | Self::Device { span, .. }
            | Self::RecordLit { span, .. }
            | Self::FieldAccess { span, .. }
            | Self::If { span, .. }
            | Self::While { span, .. }
            | Self::Break { span, .. }
            | Self::Continue { span }
            | Self::FnDef { span, .. }
            | Self::Return { span, .. } => *span,
        }
    }
}
