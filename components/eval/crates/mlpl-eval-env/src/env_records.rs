//! Saga 33 step 003: record-binding methods extracted from
//! `env.rs`. Saga 29 step 001 added the `records` map as a
//! sibling to `vars` / `strings` / `models` so the
//! one-Value-variant-per-map pattern holds.

use std::collections::BTreeMap;

use crate::env::Environment;
use mlpl_eval_types::Value;

impl Environment {
    /// Saga 29 step 001: bind a record value.
    pub fn set_record(&mut self, name: String, fields: BTreeMap<String, Value>) {
        self.records.insert(name, fields);
    }

    /// Saga 29 step 001: look up a record by name.
    #[must_use]
    pub fn get_record(&self, name: &str) -> Option<&BTreeMap<String, Value>> {
        self.records.get(name)
    }
}
