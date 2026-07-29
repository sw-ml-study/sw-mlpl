//! The `u:` function DEFINITION capability: attach pending source,
//! define, and look up user functions. The `UserFn` type itself
//! lives in mlpl-eval-state (it is an `Environment` field type).

use mlpl_eval_env::Environment;
use mlpl_eval_state::UserFn;

/// Define/look up `u:` functions and stage the raw program text so
/// `def u:` captures its span verbatim.
pub trait EnvUserFns {
    /// Attach (or clear) the raw program text for the CURRENT eval
    /// so `def u:` captures its span verbatim -- entry points that
    /// cannot use `eval_source_value` call this around their eval.
    fn set_pending_source(&mut self, src: Option<String>);
    fn define_fn(&mut self, name: String, f: UserFn);
    fn get_fn(&self, name: &str) -> Option<&UserFn>;
}

impl EnvUserFns for Environment {
    fn set_pending_source(&mut self, src: Option<String>) {
        self.pending_source = src;
    }

    fn define_fn(&mut self, name: String, f: UserFn) {
        self.user_fns.insert(name, f);
    }

    fn get_fn(&self, name: &str) -> Option<&UserFn> {
        self.user_fns.get(name)
    }
}
