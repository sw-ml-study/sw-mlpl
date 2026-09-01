//! Serde types for a catalog TOML file: a list of `[[group]]` tables,
//! each with a `name` and an `entries` array of `[name, signature,
//! doc]` triples.

use serde::Deserialize;

#[derive(Deserialize)]
pub struct Doc {
    pub group: Vec<Group>,
}

#[derive(Deserialize)]
pub struct Group {
    pub name: String,
    pub entries: Vec<(String, String, String)>,
}
