//! `HasVars`: the DenseArray-valued variable bindings. Most
//! evaluator reads hit this -- `env.get("x")` to look up a
//! value the user just bound.

use mlpl_array::DenseArray;

pub trait HasVars {
    fn get(&self, name: &str) -> Option<&DenseArray>;
    fn set(&mut self, name: String, value: DenseArray);
}
