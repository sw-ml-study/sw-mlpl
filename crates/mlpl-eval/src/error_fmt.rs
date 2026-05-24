//! Per-arm helpers for `Display for EvalError`. Saga 33 step
//! 020 split out of `error.rs` so the top-level `fmt` body
//! stays under the 50-LOC FAIL line. Grouped by the shape of
//! the variant (simple vs mismatch-triple vs record-style)
//! so each helper takes ~3 arms.

use crate::error::EvalError;

/// Format the simple-message variants: literal-text + the
/// 1-arg variants that just write a single interpolated
/// field. Returns Some(result) if `e` matched; None
/// otherwise.
pub(crate) fn fmt_simple(
    e: &EvalError,
    f: &mut std::fmt::Formatter<'_>,
) -> Option<std::fmt::Result> {
    let res = match e {
        EvalError::EmptyInput => write!(f, "empty input"),
        EvalError::UndefinedVariable(name) => write!(f, "undefined variable: {name}"),
        EvalError::InvalidRepeatCount => write!(f, "repeat count must be a scalar integer"),
        EvalError::InvalidShapeDim => {
            write!(f, "shape dimension must be a non-negative scalar integer")
        }
        EvalError::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        EvalError::ArrayError(e) => write!(f, "array error: {e}"),
        EvalError::RuntimeError(e) => write!(f, "{e}"),
        EvalError::ExpectedArray => write!(f, "expected an array value, got a string"),
        EvalError::ExpectedString => write!(f, "expected a string value"),
        EvalError::VizError(e) => write!(f, "{e}"),
        EvalError::Cancelled { step, .. } => write!(f, "cancelled at step {step}"),
        EvalError::UnwrapOnErr { message } => write!(f, "unwrap on an Err value: {message}"),
        EvalError::BreakSignal(_) | EvalError::ContinueSignal => {
            write!(f, "internal: loop-control signal escaped (bug)")
        }
        EvalError::LoopControlOutsideLoop { kind } => {
            write!(f, "{kind} used outside of a while loop")
        }
        _ => return None,
    };
    Some(res)
}

/// Format the mismatch-triple variants: ops with structured
/// (op, expected, actual) tuples.
pub(crate) fn fmt_mismatch(
    e: &EvalError,
    f: &mut std::fmt::Formatter<'_>,
) -> Option<std::fmt::Result> {
    let res = match e {
        EvalError::DeviceTensorFault { peer, device } => write!(
            f,
            "tensor lives on {peer}:{device}; use to_device('cpu', x) to fetch"
        ),
        EvalError::BadArity {
            func,
            expected,
            got,
        } => write!(f, "{func} expects {expected} arguments, got {got}"),
        EvalError::ShapeMismatch {
            op,
            expected,
            actual,
        } => write!(f, "{op}: expected {expected}, got {actual}"),
        EvalError::DeviceMismatch {
            op,
            expected,
            actual,
        } => write!(f, "device mismatch: {op} on {expected} vs {actual}"),
        EvalError::TypeMismatch {
            op,
            expected,
            actual,
            hint,
        } => write!(
            f,
            "type mismatch in {op}: expected {expected}, got {actual}\n  hint: {hint}"
        ),
        _ => return None,
    };
    Some(res)
}

/// Format the record-style variants: field-access / array-lit
/// failures and Result-style errors.
pub(crate) fn fmt_record(
    e: &EvalError,
    f: &mut std::fmt::Formatter<'_>,
) -> Option<std::fmt::Result> {
    let res = match e {
        EvalError::FieldNotFound {
            requested,
            available,
        } => write!(
            f,
            "record has no field '{requested}'; available: [{}]",
            available.join(", ")
        ),
        EvalError::FieldOnNonRecord {
            receiver_kind,
            field,
        } => write!(
            f,
            "field access '.{field}' requires a record receiver, got {receiver_kind}"
        ),
        EvalError::MixedArrayLitElements { kinds } => write!(
            f,
            "[...] array literal must be all-strings or all-numbers; got mixed kinds: [{}]",
            kinds.join(", ")
        ),
        EvalError::NotAResult {
            receiver_kind,
            accessor,
        } => write!(
            f,
            "{accessor}: expected a Result value, got {receiver_kind}"
        ),
        _ => return None,
    };
    Some(res)
}
