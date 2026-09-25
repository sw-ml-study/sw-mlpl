//! `clear_binding`: remove a name from EVERY scope table so a fresh
//! binding shadows the old KIND everywhere (lookup order must never
//! resurrect a stale binding). Per-call scoping itself -- undoing a `u:`
//! frame's writes on return -- is the undo-log journal in
//! `mlpl_eval_env::trait_impls_data`; `clear_binding` journals the name first, and
//! both expand the one table list `mlpl_eval_env::for_each_scope_table!`.

use mlpl_eval_env::Environment;

/// Kind-shadowing removal of a binding.
pub trait EnvScope {
    fn clear_binding(&mut self, name: &str);
}

impl EnvScope for Environment {
    fn clear_binding(&mut self, name: &str) {
        // Journal first: every assignment and parameter bind clears the
        // name before setting it, so this one hook covers them all.
        self.note_write(name);
        macro_rules! clear {
            ($f:ident) => {
                self.$f.remove(name)
            };
        }
        mlpl_eval_env::for_each_scope_table!(clear);
    }
}
