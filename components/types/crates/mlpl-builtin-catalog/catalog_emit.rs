//! Render a parsed catalog doc into the Rust source of its
//! `pub(crate) const GROUPS: &[FnGroup] = &[...];`. The `{:?}` on each
//! string yields a valid Rust string literal, so the generated file
//! compiles as a plain const.

use crate::catalog_parse::{Doc, Group};

/// The full `const GROUPS` definition for one domain's doc.
pub fn render(doc: &Doc) -> String {
    let groups: String = doc.group.iter().map(render_group).collect();
    format!("pub(crate) const GROUPS: &[crate::FnGroup] = &[{groups}];\n")
}

/// One `("Group name", &[(name, sig, doc), ...])` tuple.
fn render_group(g: &Group) -> String {
    let entries: String = g
        .entries
        .iter()
        .map(|(n, s, d)| format!("({n:?},{s:?},{d:?}),"))
        .collect();
    format!("({:?},&[{entries}]),", g.name)
}
