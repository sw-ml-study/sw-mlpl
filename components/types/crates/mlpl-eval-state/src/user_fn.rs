//! `UserFn` (a stored `def u:` definition) and `TestEntry` (the
//! `@test` registry row) -- split from `run_state.rs` for the
//! module function budget.

use mlpl_parser::Expr;

/// A user-defined `u:` function: params, body, the doc-string
/// convention (leading string literal), and the verbatim `def`
/// source when the entry point supplied it (`:list` prefers it).
#[derive(Clone, Debug)]
pub struct UserFn {
    pub params: Vec<String>,
    pub body: Vec<Expr>,
    pub doc: Option<String>,
    /// Verbatim `def ... }` text (comments intact) when the entry
    /// point supplied the program source; `:list` prefers this.
    pub source: Option<String>,
    /// Stacked `@word [payload]` annotations from the definition,
    /// in source order (payloads are unevaluated literal exprs;
    /// the general annotation namespace).
    pub annotations: Vec<(String, Option<Expr>)>,
}

/// One `@test`-registered function: the source-ordered registry
/// row reflection exposes (`tests()` / `test_info`).
#[derive(Clone, Debug)]
pub struct TestEntry {
    /// Stable test name (metadata `name` or the bare fn name).
    pub name: String,
    /// Full `u:`-prefixed function name (the callable key).
    pub fn_name: String,
    /// Metadata tags (empty when absent).
    pub tags: Vec<String>,
    /// Skip reason ("" = not skipped).
    pub skip: String,
    /// 1.0 when a failure is the expected outcome.
    pub expected_failure: f64,
    /// Recorded timeout policy (0 = none); ENFORCEMENT is the
    /// runner's job.
    pub timeout_ms: f64,
    /// Display name of the defining source ("repl" outside
    /// script chunks).
    pub source: String,
    /// 1-based line of the `def` in that source.
    pub line: usize,
}

impl UserFn {
    #[must_use]
    pub fn new(params: Vec<String>, body: Vec<Expr>) -> Self {
        let doc = match body.first() {
            Some(Expr::StrLit(s, _)) => Some(s.clone()),
            _ => None,
        };
        Self {
            params,
            body,
            doc,
            source: None,
            annotations: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_source(mut self, source: Option<String>) -> Self {
        self.source = source;
        self
    }

    #[must_use]
    pub fn with_annotations(mut self, annotations: Vec<(String, Option<Expr>)>) -> Self {
        self.annotations = annotations;
        self
    }

    #[must_use]
    pub fn body_exprs(&self) -> &[Expr] {
        if self.doc.is_some() && self.body.len() > 1 {
            &self.body[1..]
        } else {
            &self.body
        }
    }
}
