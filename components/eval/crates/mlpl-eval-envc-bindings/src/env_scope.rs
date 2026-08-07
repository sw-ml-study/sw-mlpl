//! Per-call variable scope (issue #6 / C1). `snapshot_scope` captures
//! the variable namespaces a user-function body may write;
//! `restore_scope` rolls them back so a call's locals (and rebound
//! params) do not leak into the caller or sibling/recursive frames.
//! Reads of outer variables still resolve against the live env while
//! the body runs.

use std::collections::{BTreeMap, HashMap};

use mlpl_array::DenseArray;
use mlpl_eval_types::Value;

use mlpl_eval_env::Environment;

pub struct ScopeSnapshot {
    vars: HashMap<String, DenseArray>,
    strings: HashMap<String, String>,
    records: HashMap<String, BTreeMap<String, Value>>,
    string_lists: HashMap<String, Vec<String>>,
    // Results were MISSING from the frame snapshot: a u: call
    // taking `ok(...)` arguments leaked its parameter bindings
    // into the caller forever (mlplunit sequencing bug,
    // 2026-08-05).
    results: HashMap<String, (bool, Value)>,
    // Function references bind into builtin_refs (both kinds), so
    // reference-valued params need the same frame restore.
    builtin_refs: HashMap<String, String>,
    partials: HashMap<String, (String, usize, Vec<Value>)>,
}

/// Per-call scope snapshot/restore for `u:` function frames, plus
/// `clear_binding`: removing a name from EVERY value table so a
/// fresh binding shadows the old KIND everywhere (lookup order
/// must never resurrect a stale binding).
pub trait EnvScope {
    #[must_use]
    fn snapshot_scope(&self) -> ScopeSnapshot;
    fn restore_scope(&mut self, s: ScopeSnapshot);
    fn clear_binding(&mut self, name: &str);
}

impl EnvScope for Environment {
    fn snapshot_scope(&self) -> ScopeSnapshot {
        ScopeSnapshot {
            vars: self.vars.clone(),
            strings: self.strings.clone(),
            records: self.records.clone(),
            string_lists: self.string_lists.clone(),
            results: self.results.clone(),
            builtin_refs: self.builtin_refs.clone(),
            partials: self.partials.clone(),
        }
    }

    fn restore_scope(&mut self, s: ScopeSnapshot) {
        self.vars = s.vars;
        self.strings = s.strings;
        self.records = s.records;
        self.string_lists = s.string_lists;
        self.results = s.results;
        self.builtin_refs = s.builtin_refs;
        self.partials = s.partials;
    }

    fn clear_binding(&mut self, name: &str) {
        self.vars.remove(name);
        self.strings.remove(name);
        self.records.remove(name);
        self.string_lists.remove(name);
        self.results.remove(name);
        self.models.remove(name);
        self.tokenizers.remove(name);
        self.gen_states.remove(name);
        self.partials.remove(name);
        self.builtin_refs.remove(name);
        self.device_tensors.remove(name);
    }
}
