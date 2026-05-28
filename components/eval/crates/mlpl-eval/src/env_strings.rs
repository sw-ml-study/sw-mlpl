//! Saga 33 step 003: string-binding methods extracted from
//! `env.rs`. Thin `HashMap` accessors; the `strings` map is
//! distinct from `vars` (Value-typed) and is used by tokenizer
//! / dataset name-bindings introduced in Saga 12 step 009.

use crate::env::Environment;

impl Environment {
    /// Bind a string value to `name`. Saga 12 step 009.
    pub fn set_string(&mut self, name: String, value: String) {
        self.strings.insert(name, value);
    }

    /// Look up a string binding by name.
    #[must_use]
    pub fn get_string(&self, name: &str) -> Option<&String> {
        self.strings.get(name)
    }
}
