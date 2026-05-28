//! Help / reference completeness checks.
//!
//! Catches doc drift the moment a builtin lands undocumented.
//! Four checks:
//!
//! 1. Every name in `BUILTIN_GROUPS` (the table backing
//!    `:builtins` in the REPL) appears at least once in
//!    `docs/lang-reference.md`.
//! 2. `BUILTIN_GROUPS` itself contains no duplicate names
//!    (would shadow earlier entries in `:describe <name>`
//!    builtin lookup).
//! 3. Every runtime dispatch arm (aggregated by
//!    `mlpl_runtime::runtime_builtin_names`) appears in
//!    `BUILTIN_GROUPS`. Without this, a new pure-array
//!    builtin can land in `mlpl-runtime` without showing up
//!    in `:builtins` or `:describe <name>`.
//! 4. `runtime_builtin_names` itself contains no duplicates
//!    across modules (e.g. `dot` accidentally listed in two
//!    `NAMES` constants would silently mean the second
//!    module's dispatch is dead code).
//!
//! Eval-side builtins (`grad`, `freeze`, `params`, `model`
//! ctors, etc.) are dispatched in `mlpl-eval` before falling
//! through to `mlpl-runtime`, so they are *not* part of the
//! runtime names list. Their coverage is enforced by check #1
//! against `BUILTIN_GROUPS`.
//!
//! Failures collect every miss into a single panic so a single
//! refactor surfaces the full set instead of failing fast.

use std::collections::HashSet;

const LANG_REFERENCE: &str = include_str!("../../../../../docs/lang-reference.md");

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
fn every_runtime_builtin_is_in_builtin_groups() {
    let documented: HashSet<&'static str> = mlpl_eval::documented_builtin_names().collect();
    let mut missing: Vec<&'static str> = Vec::new();
    for name in mlpl_runtime::runtime_builtin_names() {
        if !documented.contains(name) {
            missing.push(name);
        }
    }
    if !missing.is_empty() {
        panic!(
            "{} runtime builtin(s) not in BUILTIN_GROUPS \
             (would not show up in :builtins / :describe):\n  - {}",
            missing.len(),
            missing.join("\n  - ")
        );
    }
}

#[test]
fn runtime_builtin_names_has_no_duplicates() {
    let mut seen: HashSet<&'static str> = HashSet::new();
    let mut dups: Vec<&'static str> = Vec::new();
    for name in mlpl_runtime::runtime_builtin_names() {
        if !seen.insert(name) {
            dups.push(name);
        }
    }
    assert!(
        dups.is_empty(),
        "runtime_builtin_names has duplicates across modules \
         (would silently shadow second dispatch arm): {dups:?}"
    );
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
