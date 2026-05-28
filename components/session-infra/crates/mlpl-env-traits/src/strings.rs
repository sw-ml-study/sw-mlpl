//! `HasStrings`: string-valued variable bindings. Sibling to
//! `HasVars` (DenseArray bindings) -- the `strings` map is
//! checked before `vars` so name shadowing favors string binds
//! (Saga 12 step 009).

pub trait HasStrings {
    fn set_string(&mut self, name: String, value: String);
    fn get_string(&self, name: &str) -> Option<&String>;
}
