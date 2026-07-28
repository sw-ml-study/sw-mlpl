//! Result-binding + CLI-arg accessors (moved from the hub's
//! `result_ops.rs` in env-base-out: inherent impls must live with
//! `Environment`). The Result accessor DISPATCH stays above.

use mlpl_eval_types::Value;

use mlpl_eval_env::Environment;

/// Result bindings + CLI args.
pub trait EnvResults {
    fn set_result(&mut self, name: String, ok: bool, payload: Value);
    #[must_use]
    fn get_result(&self, name: &str) -> Option<&(bool, Value)>;
    /// Set trailing CLI args visible to `args()`. Saga 31 step 003.
    fn set_cli_args(&mut self, args: Vec<String>);
}

impl EnvResults for Environment {
    fn set_result(&mut self, name: String, ok: bool, payload: Value) {
        self.results.insert(name, (ok, payload));
    }
    fn get_result(&self, name: &str) -> Option<&(bool, Value)> {
        self.results.get(name)
    }
    fn set_cli_args(&mut self, args: Vec<String>) {
        self.cli_args = args;
    }
}
