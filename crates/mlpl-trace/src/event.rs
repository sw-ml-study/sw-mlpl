//! Trace event type.

use mlpl_core::{Span, ValueTag};
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
    /// Saga 23 step 007: per-input ValueTag snapshots. One entry
    /// per input. `None` means the corresponding input had no
    /// tag at the time of the event. Empty vec means the
    /// producer did not look up input types (e.g. literals,
    /// non-Assign expressions). Omitted from JSON when empty
    /// so untagged programs serialize unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_types: Vec<Option<ValueTag>>,
    /// Saga 23 step 007: ValueTag attached to the output, if
    /// any. Omitted from JSON when None so untagged programs
    /// serialize unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_type: Option<ValueTag>,
}
