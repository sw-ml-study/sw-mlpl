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
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty input"),
            Self::UndefinedVariable(name) => write!(f, "undefined variable: {name}"),
            Self::InvalidRepeatCount => write!(f, "repeat count must be a scalar integer"),
            Self::InvalidShapeDim => {
                write!(f, "shape dimension must be a non-negative scalar integer")
            }
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::ArrayError(e) => write!(f, "array error: {e}"),
            Self::RuntimeError(e) => write!(f, "{e}"),
            Self::ExpectedArray => write!(f, "expected an array value, got a string"),
            Self::ExpectedString => write!(f, "expected a string value"),
            Self::DeviceTensorFault { peer, device } => {
                write!(
                    f,
                    "tensor lives on {peer}:{device}; use to_device('cpu', x) to fetch"
                )
            }
            Self::BadArity {
                func,
                expected,
                got,
            } => write!(f, "{func} expects {expected} arguments, got {got}"),
            Self::VizError(e) => write!(f, "{e}"),
            Self::ShapeMismatch {
                op,
                expected,
                actual,
            } => write!(f, "{op}: expected {expected}, got {actual}"),
            Self::DeviceMismatch {
                op,
                expected,
                actual,
            } => write!(f, "device mismatch: {op} on {expected} vs {actual}"),
            Self::TypeMismatch {
                op,
                expected,
                actual,
                hint,
            } => write!(
                f,
                "type mismatch in {op}: expected {expected}, got {actual}\n  hint: {hint}"
            ),
            Self::Cancelled { step, .. } => write!(f, "cancelled at step {step}"),
            Self::FieldNotFound {
                requested,
                available,
            } => write!(
                f,
                "record has no field '{requested}'; available: [{}]",
                available.join(", ")
            ),
            Self::FieldOnNonRecord {
                receiver_kind,
                field,
            } => write!(
                f,
                "field access '.{field}' requires a record receiver, got {receiver_kind}"
            ),
            Self::MixedArrayLitElements { kinds } => write!(
                f,
                "[...] array literal must be all-strings or all-numbers; got mixed kinds: [{}]",
                kinds.join(", ")
            ),
        }
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
