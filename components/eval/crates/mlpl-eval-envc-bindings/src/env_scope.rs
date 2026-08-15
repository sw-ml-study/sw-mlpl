//! Per-call variable scope (issue #6 / C1). `snapshot_scope` captures
//! the variable namespaces a user-function body may write;
//! `restore_scope` rolls them back so a call's locals (and rebound
//! params) do not leak into the caller or sibling/recursive frames.
//! Reads of outer variables still resolve against the live env while
//! the body runs.

use std::collections::{BTreeMap, HashMap};

use mlpl_array::DenseArray;
use mlpl_eval_core::model::ModelSpec;
use mlpl_eval_core::{GenState, TokenizerSpec};
use mlpl_eval_types::Value;

use mlpl_eval_env::Environment;

#[derive(Default)]
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
    // models / tokenizers / gen_states / device_tensors were ALSO
    // missing: `clear_binding` wipes them, so a user fn whose LOCAL
    // variable shadows a global model name (e.g. `u:encode` doing
    // `m = eq(...)` while a global model `m` exists) destroyed the
    // global -- the model vanished after the call (connect MLX
    // tic-tac-toe: `undefined variable: m`). Snapshot every table
    // `clear_binding` touches so a frame fully rolls back.
    models: HashMap<String, ModelSpec>,
    tokenizers: HashMap<String, TokenizerSpec>,
    gen_states: HashMap<String, GenState>,
    device_tensors: HashMap<String, Value>,
    bytes: HashMap<String, Value>,
    ext_handles: HashMap<String, Value>,
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

/// The scope tables a `u:` frame must snapshot / restore, and that
/// `clear_binding` wipes. Listed ONCE here and expanded by all three
/// methods so they can never drift out of lockstep (define once,
/// invoke many).
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

impl EnvScope for Environment {
    fn snapshot_scope(&self) -> ScopeSnapshot {
        let mut s = ScopeSnapshot::default();
        macro_rules! snap {
            ($f:ident) => {
                s.$f = self.$f.clone()
            };
        }
        for_each_scope_table!(snap);
        s
    }

    fn restore_scope(&mut self, s: ScopeSnapshot) {
        macro_rules! restore {
            ($f:ident) => {
                self.$f = s.$f
            };
        }
        for_each_scope_table!(restore);
    }

    fn clear_binding(&mut self, name: &str) {
        macro_rules! clear {
            ($f:ident) => {
                self.$f.remove(name)
            };
        }
        for_each_scope_table!(clear);
    }
}
