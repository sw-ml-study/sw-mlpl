//! Lowering data types: the error type, the run configuration, and
//! the per-run lowering context. (lib.rs is a facade; behaviour
//! lives in the named modules.)

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

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
/// validates), the configured runtime path, and the set of
/// variables already `let`-declared in the CURRENT scope. The
/// `declared` set drives mutable bindings (first assign -> `let
/// mut`, a rebind -> reassignment, so loop accumulators mutate); it
/// is swapped for a fresh set inside a user-function body
/// (`with_scope`) and shared into `if`/`while`/`for` bodies.
pub(crate) struct Ctx {
    pub(crate) known_labels: HashMap<String, Vec<Option<String>>>,
    pub(crate) rt: TokenStream,
    pub(crate) declared: RefCell<HashSet<String>>,
    /// Bare names (no `u:` prefix) of user functions whose body
    /// produces a `CVal` (uses `ok`/`err`/`?` or returns a record),
    /// so their lowered signature is `-> CVal` and their call sites
    /// are treated as `CVal`-valued. Filled by a pre-pass before any
    /// body is lowered, so call-before-def resolves correctly.
    pub(crate) cval_returning: HashSet<String>,
    /// True while lowering the body of a `CVal`-returning function:
    /// `check` may `return` the propagated err, and the body's tail /
    /// `return` values wrap into `CVal`. Set by `with_scope`.
    pub(crate) in_cval_fn: Cell<bool>,
    /// Names (current scope) whose `let` binding holds a `CVal` rather
    /// than a `DenseArray` -- e.g. `b = read_bytes(p)?` or `s =
    /// "text"`. Lets a DenseArray-position use of the name insert the
    /// `CVal -> &DenseArray` bridge (`lower_darr`), and a CVal-typed
    /// final binding yield itself rather than a double `CVal::Arr`.
    /// Scope-swapped alongside `declared` in `with_scope`.
    pub(crate) cval_bindings: RefCell<HashSet<String>>,
}

impl Ctx {
    pub(crate) fn new(cfg: &LowerConfig) -> Self {
        Self {
            known_labels: HashMap::new(),
            rt: cfg.rt_path.clone(),
            declared: RefCell::new(HashSet::new()),
            cval_returning: HashSet::new(),
            in_cval_fn: Cell::new(false),
            cval_bindings: RefCell::new(HashSet::new()),
        }
    }

    /// Run `f` with a FRESH declared-scope pre-seeded with `names`
    /// (a function's parameters) and the given `CVal`-return mode,
    /// restoring the caller's scope + mode afterwards -- so a function
    /// body's locals never leak to, or alias, the enclosing program's
    /// variables, and `check`/tail-wrapping only fire inside a
    /// `CVal`-returning body.
    pub(crate) fn with_scope<T>(
        &self,
        names: &[String],
        returns_cval: bool,
        f: impl FnOnce() -> T,
    ) -> T {
        let fresh: HashSet<String> = names.iter().cloned().collect();
        let saved = self.declared.replace(fresh);
        let saved_mode = self.in_cval_fn.replace(returns_cval);
        // A fresh CVal-binding scope: parameters are DenseArray-valued,
        // so none start as CVal. Restored with the caller's scope.
        let saved_cval = self.cval_bindings.replace(HashSet::new());
        let out = f();
        self.declared.replace(saved);
        self.in_cval_fn.set(saved_mode);
        self.cval_bindings.replace(saved_cval);
        out
    }

    /// True the FIRST time `name` is assigned in the current scope
    /// (needs `let mut`); false on a later rebind (reassignment).
    pub(crate) fn first_binding(&self, name: &str) -> bool {
        self.declared.borrow_mut().insert(name.to_string())
    }

    /// Record whether `name`'s binding holds a `CVal` (vs a
    /// `DenseArray`), so a later use bridges/yields it correctly. A
    /// rebind to a non-CVal value clears the flag.
    pub(crate) fn set_cval_binding(&self, name: &str, is_cval: bool) {
        let mut bindings = self.cval_bindings.borrow_mut();
        if is_cval {
            bindings.insert(name.to_string());
        } else {
            bindings.remove(name);
        }
    }
}
