//! Short machine tags for `EvalError` variants (spike step 011).
//!
//! `try { } catch e { }` binds `e` to the canonical error record
//! `{kind, message}` (the Dyalog Quad-DMX move, see
//! docs/error-handling.md); `kind` comes from here so handlers can
//! dispatch on `e.kind` without parsing prose.

use crate::error::EvalError;

/// Kebab-case machine tag for one error variant. Stable across
/// message-wording changes; grouped the way a HANDLER would branch
/// (shape-ish problems -> "shape", arity -> "arity", etc.).
#[must_use]
pub fn error_kind(e: &EvalError) -> &'static str {
    match e {
        EvalError::EmptyInput => "empty-input",
        EvalError::UndefinedVariable(_) => "undefined-variable",
        EvalError::Unsupported(_) => "unsupported",
        EvalError::ArrayError(_) | EvalError::ShapeMismatch { .. } => "shape",
        EvalError::InvalidRepeatCount | EvalError::InvalidShapeDim => "invalid-argument",
        EvalError::RuntimeError(_) => "runtime",
        EvalError::ExpectedArray | EvalError::ExpectedString | EvalError::TypeMismatch { .. } => {
            "type"
        }
        EvalError::DeviceTensorFault { .. } | EvalError::DeviceMismatch { .. } => "device",
        EvalError::BadArity { .. } => "arity",
        EvalError::VizError(_) => "viz",
        EvalError::Cancelled { .. } => "cancelled",
        EvalError::FieldNotFound { .. } | EvalError::FieldOnNonRecord { .. } => "field",
        EvalError::MixedArrayLitElements { .. } => "type",
        EvalError::UnwrapOnErr { .. } => "unwrap-on-err",
        EvalError::ExitRequested(_) => "exit",
        EvalError::NotAResult { .. } => "not-a-result",
        EvalError::BreakSignal(_)
        | EvalError::ContinueSignal
        | EvalError::ReturnSignal(_)
        | EvalError::LoopControlOutsideLoop { .. } => "control-flow",
    }
}
