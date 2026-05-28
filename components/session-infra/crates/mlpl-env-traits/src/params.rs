//! `HasParams` + `HasFrozen`: the trainable-parameter
//! machinery. Bundled into one file so the crate stays under
//! the 7-module Crate-Module-Count FAIL line.
//!
//! `HasParams`: trainable parameter bindings. Distinct from
//! `HasVars` so a consumer can write functions that only touch
//! parameters (e.g. the optimizer step) without holding a
//! reference that could mutate user-bound variables.
//!
//! `HasFrozen`: the frozen-parameter set. Saga 15 step 001
//! introduced the set so `adam` / `momentum_sgd` skip names in
//! it at the optimizer update step -- gradients still flow,
//! but the parameter update is suppressed.

use mlpl_array::DenseArray;

pub trait HasParams {
    /// Bind `name` to a trainable parameter value. Records the
    /// name in the params set so `is_param` returns true.
    fn set_param(&mut self, name: String, value: DenseArray);

    /// Whether `name` is currently in the params set.
    fn is_param(&self, name: &str) -> bool;

    /// Add `name` to the params set without binding a value.
    /// Used by model constructors that want to declare a name
    /// as trainable before its initial value lands.
    fn mark_param(&mut self, name: &str);
}

pub trait HasFrozen {
    /// Add `name` to the frozen set.
    fn mark_frozen(&mut self, name: &str);

    /// Remove `name` from the frozen set. No-op if not present.
    fn unmark_frozen(&mut self, name: &str);

    /// Whether `name` is currently frozen.
    fn is_frozen(&self, name: &str) -> bool;
}
