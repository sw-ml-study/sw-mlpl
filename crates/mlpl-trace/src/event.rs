//! Trace event type.

use mlpl_core::Span;
use serde::{Deserialize, Serialize};

use crate::value::TraceValue;

/// A single recorded evaluation step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceEvent {
    /// Sequence number (auto-incrementing).
    pub seq: u64,
    /// Operation description (e.g., "add", "literal", "reshape").
    pub op: String,
    /// Source location.
    pub span: Span,
    /// Input value snapshots.
    pub inputs: Vec<TraceValue>,
    /// Output value snapshot.
    pub output: TraceValue,
}
