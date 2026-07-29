//! Saga 33 step 002: frozen-parameter methods extracted
//! from `env.rs`. Saga 15 step 001 added the `frozen_params`
//! `HashSet` so `adam` / `momentum_sgd` skip updates for names
//! in the set -- the user-facing surface for "freeze the
//! pretrained base, train only the `LoRA` adapter."

use mlpl_eval_env::Environment;

/// Frozen-parameter set (optimizer-update suppression).
pub trait EnvFrozen {
    /// Saga 15 step 001: mark `name` as frozen. `adam` and
    /// `momentum_sgd` skip any frozen name when applying
    /// parameter updates.
    fn mark_frozen(&mut self, name: &str);
    /// Saga 15 step 001: remove `name` from the frozen set.
    /// No-op if `name` is not currently frozen.
    fn unmark_frozen(&mut self, name: &str);
    /// Saga 15 step 001: whether `name` is currently frozen.
    #[must_use]
    fn is_frozen(&self, name: &str) -> bool;
}

impl EnvFrozen for Environment {
    fn mark_frozen(&mut self, name: &str) {
        self.frozen_params.insert(name.to_string());
    }

    fn unmark_frozen(&mut self, name: &str) {
        self.frozen_params.remove(name);
    }

    fn is_frozen(&self, name: &str) -> bool {
        self.frozen_params.contains(name)
    }
}
