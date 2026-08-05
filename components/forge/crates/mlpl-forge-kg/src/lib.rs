//! Knowledge-graph task oracle (docs/data-forge-design.md). A
//! graph is plain data: entities are integer ids and the graph is
//! an `[E, 3]` edge array of `(src, relation, dst)` rows. These
//! four builtins make it an ORACLE for multi-hop reasoning tasks:
//! generate valid paths, verify candidate paths, walk one-hop
//! neighborhoods, and split by entity so eval visits graph
//! regions training never saw.

mod edges;
mod ops;
mod paths;

pub use ops::try_call;
pub use paths::split_edges;

/// Builtin names this crate dispatches (chained into the runtime
/// NAMES list for help completeness).
pub const NAMES: &[&str] = &["kg_neighbors", "kg_verify", "kg_paths", "kg_split"];
