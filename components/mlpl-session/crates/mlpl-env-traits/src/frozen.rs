//! `HasFrozen`: the frozen-parameter set. Saga 15 step 001
//! introduced the set so `adam` / `momentum_sgd` skip names in
//! it at the optimizer update step -- gradients still flow,
//! but the parameter update is suppressed. User-facing surface
//! for "freeze the pretrained base, train only the LoRA
//! adapter."

pub trait HasFrozen {
    /// Add `name` to the frozen set.
    fn mark_frozen(&mut self, name: &str);

    /// Remove `name` from the frozen set. No-op if not present.
    fn unmark_frozen(&mut self, name: &str);

    /// Whether `name` is currently frozen.
    fn is_frozen(&self, name: &str) -> bool;
}
