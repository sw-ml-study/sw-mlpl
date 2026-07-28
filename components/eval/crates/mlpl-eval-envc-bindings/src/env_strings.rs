//! Saga 33 step 003: string-binding methods extracted from
//! `env.rs`. Thin `HashMap` accessors; the `strings` map is
//! distinct from `vars` (Value-typed) and is used by tokenizer
//! / dataset name-bindings introduced in Saga 12 step 009.

use mlpl_eval_env::Environment;

/// String variable bindings.
pub trait EnvStrings {
    /// Bind a string value to `name`. Saga 12 step 009.
    fn set_string(&mut self, name: String, value: String);
    /// Look up a string binding by name.
    #[must_use]
    fn get_string(&self, name: &str) -> Option<&String>;
}

impl EnvStrings for Environment {
    fn set_string(&mut self, name: String, value: String) {
        self.strings.insert(name, value);
    }

    fn get_string(&self, name: &str) -> Option<&String> {
        self.strings.get(name)
    }
}
