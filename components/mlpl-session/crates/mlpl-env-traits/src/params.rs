//! `HasParams`: trainable parameter bindings. Distinct from
//! `HasVars` so a consumer can write functions that only touch
//! parameters (e.g. the optimizer step) without holding a
//! reference that could mutate user-bound variables.

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
