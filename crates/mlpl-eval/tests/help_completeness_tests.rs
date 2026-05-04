//! Help / reference completeness checks.
//!
//! Catches doc drift the moment a builtin lands undocumented.
//! Two checks:
//!
//! 1. Every name in `BUILTIN_GROUPS` (the table backing
//!    `:builtins` in the REPL) appears at least once in
//!    `docs/lang-reference.md`.
//! 2. `BUILTIN_GROUPS` itself contains no duplicate names
//!    (would shadow earlier entries in `:describe <name>`
//!    builtin lookup).
//!
//! Failures collect every miss into a single panic so a single
//! refactor surfaces the full set instead of failing fast.
//!
//! Out of scope (deferred): reverse direction --
//! "every runtime dispatch arm appears in `BUILTIN_GROUPS`".
//! That requires aggregating the per-source-file `match name {
//! ... }` arms in `mlpl-runtime`. Once that crate exposes a
//! `runtime_builtin_names()` accessor, a sibling test fills
//! the gap.

use std::collections::HashSet;

const LANG_REFERENCE: &str = include_str!("../../../docs/lang-reference.md");

#[test]
fn every_builtin_groups_name_is_in_lang_reference() {
    let mut missing: Vec<&'static str> = Vec::new();
    for name in mlpl_eval::documented_builtin_names() {
        if !LANG_REFERENCE.contains(name) {
            missing.push(name);
        }
    }
    if !missing.is_empty() {
        panic!(
            "{} builtin name(s) listed in BUILTIN_GROUPS but \
             missing from docs/lang-reference.md:\n  - {}",
            missing.len(),
            missing.join("\n  - ")
        );
    }
}

#[test]
fn builtin_groups_has_no_duplicate_names() {
    let mut seen: HashSet<&'static str> = HashSet::new();
    let mut dups: Vec<&'static str> = Vec::new();
    for name in mlpl_eval::documented_builtin_names() {
        if !seen.insert(name) {
            dups.push(name);
        }
    }
    assert!(
        dups.is_empty(),
        "BUILTIN_GROUPS has duplicate names (would shadow in :describe lookup): {dups:?}"
    );
}
