//! Saga 33 step 003: string-list binding methods extracted from
//! `env.rs`. Saga 29 step 002 added `string_lists` as a sibling
//! map for `["cat", "dog"]`-shaped values.

use crate::env::Environment;

impl Environment {
    /// Saga 29 step 002: bind a string-list value.
    pub fn set_string_list(&mut self, name: String, items: Vec<String>) {
        self.string_lists.insert(name, items);
    }

    /// Saga 29 step 002: look up a string-list by name.
    #[must_use]
    pub fn get_string_list(&self, name: &str) -> Option<&Vec<String>> {
        self.string_lists.get(name)
    }
}
