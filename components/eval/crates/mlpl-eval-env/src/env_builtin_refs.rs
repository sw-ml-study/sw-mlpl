//! Saga 33 step 003: builtin-reference binding methods extracted
//! from `env.rs`. The `builtin_refs` map records `f = :add` ->
//! `("f", "add")`. `eval_expr(Expr::Ident("f"))` checks this
//! map after `vars` / `strings`, so user variables can't shadow
//! builtin references.

use crate::env::Environment;

impl Environment {
    pub fn set_builtin_ref(&mut self, name: String, target: String) {
        self.builtin_refs.insert(name, target);
    }

    /// Look up a builtin / operator reference by binding name.
    /// Returns the target builtin name, or `None` if `name` is
    /// not bound to a `BuiltinRef`.
    #[must_use]
    pub fn get_builtin_ref(&self, name: &str) -> Option<&String> {
        self.builtin_refs.get(name)
    }
}
