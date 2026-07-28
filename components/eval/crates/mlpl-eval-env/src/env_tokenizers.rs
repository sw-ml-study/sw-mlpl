//! Saga 33 step 003: tokenizer-registry methods extracted from
//! `env.rs`. The actual `TokenizerSpec` lives in
//! `mlpl_eval_core::TokenizerSpec`; this module is a thin `HashMap` accessor
//! layer kept separate from `env_models.rs` so each sibling stays
//! under the module-function-count warn line.

use crate::env::Environment;
use mlpl_eval_core::TokenizerSpec;

impl Environment {
    /// Bind `name` to a tokenizer value. Saga 12 step 004.
    pub fn set_tokenizer(&mut self, name: String, tok: TokenizerSpec) {
        self.tokenizers.insert(name, tok);
    }

    /// Look up a tokenizer by name. Returns `None` if `name` is
    /// not bound to a tokenizer.
    #[must_use]
    pub fn get_tokenizer(&self, name: &str) -> Option<&TokenizerSpec> {
        self.tokenizers.get(name)
    }

    /// Iterate over every bound `(name, TokenizerSpec)`. Saga 21
    /// step 002: same use case as `models_iter`.
    pub fn tokenizers_iter(&self) -> impl Iterator<Item = (&String, &TokenizerSpec)> {
        self.tokenizers.iter()
    }
}
