//! `LearningPath` + `Step` enum -- the shared data model
//! consumed by the per-theme path constants in
//! `history` / `skills` / `architectures` / `visual`.

pub struct LearningPath {
    pub title: &'static str,
    pub blurb: &'static str,
    pub steps: &'static [Step],
}

#[derive(Clone, Copy, PartialEq)]
pub enum Step {
    /// A tutorial lesson, looked up by exact title.
    Lesson {
        title: &'static str,
        why: &'static str,
    },
    /// A demo, looked up by exact name.
    Demo {
        name: &'static str,
        why: &'static str,
    },
    /// A diagram, looked up by filename slug (matching the
    /// numbered `<slug>.svg` files in `diagrams/`).
    Diagram {
        slug: &'static str,
        why: &'static str,
    },
    /// A glossary entry, looked up by exact term (matching
    /// `## TermName` headers in `docs/glossary.md`).
    Glossary {
        term: &'static str,
        why: &'static str,
    },
    /// A path-orientation note that does not reference
    /// existing content. Shown as a small framing card.
    Note {
        title: &'static str,
        body: &'static str,
    },
}
