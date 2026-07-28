//! Saga 33 step 013: `impl HasStrings for Environment`.

use mlpl_env_traits::HasStrings;

use crate::env::Environment;

impl HasStrings for Environment {
    fn set_string(&mut self, name: String, value: String) {
        self.strings.insert(name, value);
    }
    fn get_string(&self, name: &str) -> Option<&String> {
        self.strings.get(name)
    }
}
