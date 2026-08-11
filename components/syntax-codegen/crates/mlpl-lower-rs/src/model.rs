//! Lowering data types: the error type, the run configuration, and
//! the per-run lowering context. (lib.rs is a facade; behaviour
//! lives in the named modules.)

use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::quote;

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

/// Per-run lowering context: compile-time knowledge of a binding's
/// axis labels (built up as top-level statements are walked; a
/// missing name skips the static check, the runtime still
/// validates) plus the configured runtime path.
pub(crate) struct Ctx {
    pub(crate) known_labels: HashMap<String, Vec<Option<String>>>,
    pub(crate) rt: TokenStream,
}

impl Ctx {
    pub(crate) fn new(cfg: &LowerConfig) -> Self {
        Self {
            known_labels: HashMap::new(),
            rt: cfg.rt_path.clone(),
        }
    }
}
