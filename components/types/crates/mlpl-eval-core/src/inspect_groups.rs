//! Facade: the curated builtin catalog moved to the dedicated
//! `mlpl-builtin-catalog` crate (static data split by domain);
//! this module keeps the long-standing import path stable.

pub use mlpl_builtin_catalog::{FnEntry, FnGroup, builtin_groups, documented_builtin_names};
