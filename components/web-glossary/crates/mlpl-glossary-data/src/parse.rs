//! Parse `docs/glossary.md` (compiled in via `include_str!`) into glossary
//! entries, and look entries up by term. Pure -- no yew/wasm.

use std::sync::OnceLock;

use crate::model::{GlossaryDoc, GlossaryEntry};

const GLOSSARY_MD: &str = include_str!("../../../../../docs/glossary.md");

/// The parsed glossary, parsed once and cached for the process.
pub fn doc() -> &'static GlossaryDoc {
    static DOC: OnceLock<GlossaryDoc> = OnceLock::new();
    DOC.get_or_init(parse_glossary)
}

/// Look up a glossary entry by exact term match (case-insensitive). Used by
/// the Paths walker to render an inline glossary excerpt for `Step::Glossary`
/// steps.
pub fn find_by_term(term: &str) -> Option<&'static GlossaryEntry> {
    let q = term.trim().to_ascii_lowercase();
    doc()
        .entries
        .iter()
        .find(|e| e.term.to_ascii_lowercase() == q)
}

/// Parse `GLOSSARY_MD` into an intro blurb + per-term entries, sorted
/// alphabetically (case-insensitive) so entries appended to glossary.md
/// still browse under the right letter regardless of file order.
fn parse_glossary() -> GlossaryDoc {
    let mut intro = String::new();
    let mut entries = Vec::new();
    let mut chunks = GLOSSARY_MD.split("\n## ");
    if let Some(first) = chunks.next() {
        intro = first
            .lines()
            .filter(|l| !l.starts_with("# ") && !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
    }
    for chunk in chunks {
        if let Some((term_line, rest)) = chunk.split_once('\n') {
            let term = term_line.trim().to_string();
            let slug = slugify(&term);
            entries.push(GlossaryEntry {
                term,
                slug,
                body: rest.trim().to_string(),
            });
        }
    }
    entries.sort_by_key(|e| e.term.to_ascii_lowercase());
    GlossaryDoc { intro, entries }
}

/// A stable, url-safe id from a term: lowercased alphanumerics, runs of other
/// characters collapsed to single dashes, trimmed.
fn slugify(term: &str) -> String {
    let mut out = String::with_capacity(term.len());
    let mut last_dash = true;
    for c in term.chars() {
        if c.is_ascii_alphanumeric() {
            out.extend(c.to_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}
