//! Curated `:builtins` / `:describe <name>` catalog. Static data
//! only -- no formatting or dispatch logic -- split by domain so
//! each half stays inside the file-size budget as builtins grow.
//!
//! Completeness tests elsewhere assert every name here appears in
//! `docs/lang-reference.md` and `docs/glossary.md`, and that every
//! runtime dispatch arm is covered here.

mod groups_lang;
mod groups_ml;

/// `(name, signature, one-line doc)` row used by both the grouped
/// `:fns` listing and the flat `:describe <builtin>` lookup.
pub type FnEntry = (&'static str, &'static str, &'static str);
/// `(group_label, entries)` used by `:fns`.
pub type FnGroup = (&'static str, &'static [FnEntry]);

/// Every curated group, language/array domains first then ML.
pub fn builtin_groups() -> &'static [FnGroup] {
    static ALL: std::sync::OnceLock<Vec<FnGroup>> = std::sync::OnceLock::new();
    ALL.get_or_init(|| {
        let mut v = Vec::new();
        v.extend_from_slice(groups_lang::GROUPS);
        v.extend_from_slice(groups_ml::GROUPS);
        v
    })
}

/// Iterate every builtin name in the catalog (docs-completeness
/// tests and the glossary coverage pin consume this).
pub fn documented_builtin_names() -> impl Iterator<Item = &'static str> {
    builtin_groups()
        .iter()
        .flat_map(|(_, entries)| entries.iter().map(|(name, _, _)| *name))
}
