//! Glossary data types.

/// One parsed glossary entry: the term, a stable id slug (for in-page
/// anchors), and the markdown body.
pub struct GlossaryEntry {
    pub term: String,
    pub slug: String,
    pub body: String,
}

/// The parsed glossary: an intro blurb plus the alphabetized entries.
pub struct GlossaryDoc {
    pub intro: String,
    pub entries: Vec<GlossaryEntry>,
}
