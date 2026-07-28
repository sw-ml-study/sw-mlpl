//! Result-binding + CLI-arg accessors (moved from the hub's
//! `result_ops.rs` in env-base-out: inherent impls must live with
//! `Environment`). The Result accessor DISPATCH stays above.

use mlpl_eval_types::Value;

use crate::env::Environment;

impl Environment {
    pub fn set_result(&mut self, name: String, ok: bool, payload: Value) {
        self.results.insert(name, (ok, payload));
    }
    #[must_use]
    pub fn get_result(&self, name: &str) -> Option<&(bool, Value)> {
        self.results.get(name)
    }
    /// Set trailing CLI args visible to `args()`. Saga 31 step 003.
    pub fn set_cli_args(&mut self, args: Vec<String>) {
        self.cli_args = args;
    }
}
