//! Per-call variable scope (issue #6 / C1). `snapshot_scope` captures
//! the variable namespaces a user-function body may write;
//! `restore_scope` rolls them back so a call's locals (and rebound
//! params) do not leak into the caller or sibling/recursive frames.
//! Reads of outer variables still resolve against the live env while
//! the body runs.

use std::collections::{BTreeMap, HashMap};

use mlpl_array::DenseArray;
use mlpl_eval_types::Value;

use crate::env::Environment;

pub struct ScopeSnapshot {
    vars: HashMap<String, DenseArray>,
    strings: HashMap<String, String>,
    records: HashMap<String, BTreeMap<String, Value>>,
    string_lists: HashMap<String, Vec<String>>,
}

impl Environment {
    #[must_use]
    pub fn snapshot_scope(&self) -> ScopeSnapshot {
        ScopeSnapshot {
            vars: self.vars.clone(),
            strings: self.strings.clone(),
            records: self.records.clone(),
            string_lists: self.string_lists.clone(),
        }
    }

    pub fn restore_scope(&mut self, s: ScopeSnapshot) {
        self.vars = s.vars;
        self.strings = s.strings;
        self.records = s.records;
        self.string_lists = s.string_lists;
    }
}
