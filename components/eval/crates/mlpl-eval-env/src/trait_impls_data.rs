//! Binding DATA for `Environment`: the `mlpl-env-traits` impls `HasVars`,
//! `HasStrings`, `HasModels`, and the undo-log frames that scope every
//! binding write inside a `u:` call.
//!
//! A frame records, the first time it writes a name, that name's prior
//! binding in every scope table; on exit it restores exactly those names.
//! A call therefore costs O(names it writes) -- not a deep copy of every
//! global, which made a call with a multi-megabyte global in scope cost
//! 200x more (microgpt-mlpl issue e). Every scope-table write goes through
//! `note_write` first; the table list is defined once in
//! `for_each_scope_table!` so capture, restore and `clear_binding` can
//! never drift apart.

use std::collections::{BTreeMap, HashMap};

use mlpl_array::DenseArray;
use mlpl_env_traits::{HasModels, HasStrings, HasVars};
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_core::{GenState, TokenizerSpec};
use mlpl_eval_types::Value;

use crate::env::Environment;

/// The value tables a `u:` frame scopes. Listed once; expanded by the
/// journal capture and restore below and by `clear_binding`
/// (mlpl-eval-envc-bindings), so no table can be missed by one of them.
#[macro_export]
macro_rules! for_each_scope_table {
    ($m:ident) => {
        $m!(vars);
        $m!(strings);
        $m!(records);
        $m!(string_lists);
        $m!(results);
        $m!(builtin_refs);
        $m!(partials);
        $m!(models);
        $m!(tokenizers);
        $m!(gen_states);
        $m!(device_tensors);
        $m!(bytes);
        $m!(ext_handles);
    };
}

/// One name's binding in every scope table (`None` = unbound there),
/// captured before a frame's first write to the name.
#[derive(Clone, Debug, Default)]
pub struct PriorBinding {
    vars: Option<DenseArray>,
    strings: Option<String>,
    records: Option<BTreeMap<String, Value>>,
    string_lists: Option<Vec<String>>,
    results: Option<(bool, Value)>,
    builtin_refs: Option<String>,
    partials: Option<(String, usize, Vec<Value>)>,
    models: Option<ModelSpec>,
    tokenizers: Option<TokenizerSpec>,
    gen_states: Option<GenState>,
    device_tensors: Option<Value>,
    bytes: Option<Value>,
    ext_handles: Option<Value>,
}

/// The names one frame has written, with their pre-frame bindings. A frame
/// opens by pushing an empty journal onto `Environment::frame_journal`.
pub type FrameJournal = HashMap<String, PriorBinding>;

impl Environment {
    /// Close the innermost frame, restoring every name it wrote to its
    /// pre-frame binding (removing names that did not exist).
    pub fn frame_exit(&mut self) {
        let Some(journal) = self.frame_journal.pop() else {
            return;
        };
        for (name, prior) in journal {
            macro_rules! restore {
                ($f:ident) => {
                    match prior.$f {
                        Some(v) => {
                            self.$f.insert(name.clone(), v);
                        }
                        None => {
                            self.$f.remove(&name);
                        }
                    }
                };
            }
            for_each_scope_table!(restore);
        }
    }

    /// Call before ANY write to a scope table under `name`: inside a frame,
    /// the first write records the name's current bindings for
    /// `frame_exit`. A no-op at top level.
    pub fn note_write(&mut self, name: &str) {
        match self.frame_journal.last() {
            Some(j) if !j.contains_key(name) => {}
            _ => return,
        }
        let mut prior = PriorBinding::default();
        macro_rules! capture {
            ($f:ident) => {
                prior.$f = self.$f.get(name).cloned();
            };
        }
        for_each_scope_table!(capture);
        if let Some(j) = self.frame_journal.last_mut() {
            j.insert(name.to_string(), prior);
        }
    }
}

impl HasVars for Environment {
    fn get(&self, name: &str) -> Option<&DenseArray> {
        self.vars.get(name)
    }
    fn set(&mut self, name: String, value: DenseArray) {
        self.note_write(&name);
        self.vars.insert(name, value);
    }
}

impl HasStrings for Environment {
    fn set_string(&mut self, name: String, value: String) {
        self.note_write(&name);
        self.strings.insert(name, value);
    }
    fn get_string(&self, name: &str) -> Option<&String> {
        self.strings.get(name)
    }
}

impl HasModels for Environment {
    fn get_model(&self, name: &str) -> Option<&ModelSpec> {
        self.models.get(name)
    }
}
