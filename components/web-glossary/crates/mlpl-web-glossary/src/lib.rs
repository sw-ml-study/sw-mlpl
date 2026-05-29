//! Glossary surface for the web playground -- the term-lookup
//! data + view (browseable docs panel tab) + the
//! click-anywhere-in-page popup host. Extracted from mlpl-web
//! during saga 82. mlpl-web re-exports the two modules under
//! their original `crate::glossary_popup` / `crate::glossary_view`
//! names so callers stay unchanged.

pub mod popup;
pub mod view;
