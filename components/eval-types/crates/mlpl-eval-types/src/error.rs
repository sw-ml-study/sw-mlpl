//! Evaluation error types.

use mlpl_core::LabeledShape;

/// Errors produced during evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum EvalError {
    /// No expressions to evaluate.
    EmptyInput,
    /// Variable not found in environment.
    UndefinedVariable(String),
    /// Feature not yet implemented.
    Unsupported(String),
    /// Error from array operations.
    ArrayError(mlpl_array::ArrayError),
    /// Repeat count must be a scalar.
    InvalidRepeatCount,
    /// Tensor constructor shape dimension must be a non-negative scalar integer.
    InvalidShapeDim,
    /// Error from built-in function dispatch.
    RuntimeError(mlpl_runtime::RuntimeError),
    /// Expected an array value but got a string.
    ExpectedArray,
    /// Expected a string value but got something else.
    ExpectedString,
    /// Saga R1 step 002: a CPU op tried to consume a
    /// `Value::DeviceTensor` whose bytes live on a
    /// peer server. Strict-fault: the user must
    /// explicitly `to_device('cpu', x)` to fetch.
    DeviceTensorFault {
        /// Peer URL the tensor lives on.
        peer: String,
        /// Device name on that peer (e.g., `"mlx"`).
        device: String,
    },
    /// Wrong number of arguments to a built-in.
    BadArity {
        /// Function name.
        func: String,
        /// Expected count.
        expected: usize,
        /// Got count.
        got: usize,
    },
    /// Error from the visualization layer.
    VizError(mlpl_viz::VizError),
    /// Two operand shapes (or labels) disagree in a way the named op
    /// cannot resolve. Saga 11.5 Phase 4: replaces string
    /// `Unsupported` messages for broadcasting and contraction
    /// failures. `expected` and `actual` are the left- and right-hand
    /// operand labeled shapes respectively.
    ShapeMismatch {
        /// Operator or builtin name (`"add"`, `"matmul"`, ...).
        op: String,
        /// Left-hand operand's labeled shape.
        expected: LabeledShape,
        /// Right-hand operand's labeled shape.
        actual: LabeledShape,
    },
    /// Two tensors (or a tensor and the active `device("...") { }`
    /// scope) disagree on device placement. Saga 14 step 005: raised
    /// by `apply(model, X)` when the input lives on a different
    /// device than the model's parameters, and by any op that
    /// receives mixed-device operands. `op` names the site
    /// (`"matmul"`, `"apply"`, `"add"`, ...); `expected` is the
    /// device the left-hand side carries and `actual` is the
    /// right-hand side's.
    DeviceMismatch {
        /// Operator or builtin name.
        op: String,
        /// Device the left-hand (or first) operand is on.
        expected: String,
        /// Device the right-hand (or second) operand is on.
        actual: String,
    },
    /// Saga 23 step 004: a typed-value consumer received an
    /// argument whose `ValueTag` does not satisfy the
    /// predicate. `expected` is the tag-display-name the
    /// consumer wants ("Logit", "Loss", ...), `actual` is the
    /// tag-display-name the argument carries, and `hint` is a
    /// 3-5 line tutoring message naming the most likely cause
    /// and one or two concrete fixes. Untagged arguments
    /// always pass and never raise this error (gradual-typing
    /// additivity).
    TypeMismatch {
        /// Consumer op name.
        op: String,
        /// Tag the consumer wants.
        expected: String,
        /// Tag the argument actually carries.
        actual: String,
        /// Tutoring hint (multi-line ASCII text).
        hint: String,
    },
    /// Saga 21.5 step 003: cooperative cancellation observed at a
    /// loop head or pre-builtin checkpoint. `step` is the
    /// inner-most loop iteration the eval was on when the trip was
    /// caught (`0` for non-loop sites). `partial_losses` is the
    /// per-iteration loss curve accumulated so far inside `train`
    /// (empty for `for` / `repeat` / pre-builtin sites). The
    /// session's `last_losses` binding is also populated with the
    /// same vector so post-cancel `:vars` still sees the partial
    /// curve.
    Cancelled {
        /// Iteration index at the trip site (0 for non-loop).
        step: usize,
        /// Per-iteration losses recorded so far inside `train`.
        partial_losses: Vec<f64>,
    },
    /// Saga 29 step 001: field access on a record that does not
    /// have the requested field. Lists the available keys so the
    /// user can fix the typo.
    FieldNotFound {
        /// Field name the user asked for.
        requested: String,
        /// Field names the record actually has, sorted.
        available: Vec<String>,
    },
    /// Saga 29 step 001: field access on a value that is not a
    /// record. The receiver_kind names the variant for the
    /// tutoring message ("array", "string", "model", etc.).
    FieldOnNonRecord {
        /// What kind of value the receiver was.
        receiver_kind: &'static str,
        /// Field name the user asked for.
        field: String,
    },
    /// Saga 29 step 002: a `[...]` array literal contained
    /// elements of more than one kind (e.g. mixing strings and
    /// numbers). The MLPL `[...]` literal is monomorphic: every
    /// element must evaluate to the same `Value` kind. `kinds`
    /// is the per-position list of `value_kind()` results, in
    /// source order, so the tutoring message can show which
    /// element broke the rule.
    MixedArrayLitElements {
        /// `value_kind()` of each element, in source order.
        kinds: Vec<&'static str>,
    },
    /// Saga 29 step 012: caller invoked `unwrap(r)` on an
    /// `Err(_)` value. The inner error payload's display
    /// form is recorded here so the user sees what went
    /// wrong rather than a generic "unwrap failed".
    UnwrapOnErr {
        /// Display-format of the Err payload.
        message: String,
    },
    /// Saga 29 step 012: caller invoked an Ok-only accessor
    /// (e.g. `unwrap`) on a non-Result value, or an Err-only
    /// accessor (`err_message`) on a non-Result.
    NotAResult {
        /// `value_kind()` of the receiver.
        receiver_kind: &'static str,
        /// Name of the accessor that was called.
        accessor: &'static str,
    },
    /// Saga 31 step 005: internal `break value` signal.
    /// Propagated by `?` out of a `while` body and caught by
    /// the loop driver. The value is boxed because
    /// `Value` is a large enum and inlining it inflates
    /// `Result<_, EvalError>` across the whole crate. If the
    /// signal escapes a `while` it becomes the
    /// `break/continue outside of a loop` error surfaced via
    /// the `LoopControlOutsideLoop` variant below.
    BreakSignal(Box<crate::value::Value>),
    /// Saga 31 step 005: internal `continue` signal.
    ContinueSignal,
    /// Saga 46: internal `return` signal from a UDF body.
    ReturnSignal(Box<crate::value::Value>),
    /// `exit(code)` under an intercept (run_script / tests):
    /// unwinds like a signal, NOT catchable by try/catch, and
    /// surfaces as structured exit status instead of killing
    /// the process.
    ExitRequested(u8),
    /// Saga 31 step 005: `break` or `continue` evaluated
    /// without an enclosing `while` loop to catch the signal.
    /// `kind` is `"break"` or `"continue"`.
    LoopControlOutsideLoop {
        /// Which keyword triggered this: `"break"` or `"continue"`.
        kind: &'static str,
    },
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Saga 33 step 020: per-arm fmt helpers live in
        // `error_fmt.rs`. Each helper returns Some(result) if
        // it matched the variant; otherwise None and we fall
        // through to the next group.
        if let Some(res) = crate::error_fmt::fmt_simple(self, f) {
            return res;
        }
        if let Some(res) = crate::error_fmt::fmt_mismatch(self, f) {
            return res;
        }
        if let Some(res) = crate::error_fmt::fmt_record(self, f) {
            return res;
        }
        unreachable!("EvalError variant not covered by fmt_simple / fmt_mismatch / fmt_record")
    }
}

impl std::error::Error for EvalError {}

impl From<mlpl_array::ArrayError> for EvalError {
    fn from(e: mlpl_array::ArrayError) -> Self {
        Self::ArrayError(e)
    }
}

impl From<mlpl_runtime::RuntimeError> for EvalError {
    fn from(e: mlpl_runtime::RuntimeError) -> Self {
        Self::RuntimeError(e)
    }
}

impl From<mlpl_viz::VizError> for EvalError {
    fn from(e: mlpl_viz::VizError) -> Self {
        Self::VizError(e)
    }
}
