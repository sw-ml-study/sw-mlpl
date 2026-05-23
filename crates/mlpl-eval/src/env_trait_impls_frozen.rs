//! Saga 33 step 010: `impl HasFrozen for Environment`.

use mlpl_env_traits::HasFrozen;

use crate::env::Environment;

impl HasFrozen for Environment {
    fn mark_frozen(&mut self, name: &str) {
        Environment::mark_frozen(self, name);
    }
    fn unmark_frozen(&mut self, name: &str) {
        Environment::unmark_frozen(self, name);
    }
    fn is_frozen(&self, name: &str) -> bool {
        Environment::is_frozen(self, name)
    }
}
