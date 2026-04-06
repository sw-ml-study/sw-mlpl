//! Trace container.

use serde::{Deserialize, Serialize};

use crate::event::TraceEvent;

/// An ordered collection of trace events for one evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    source: String,
    events: Vec<TraceEvent>,
}

impl Trace {
    /// Create a new empty trace for the given source code.
    #[must_use]
    pub fn new(source: String) -> Self {
        Self {
            source,
            events: Vec::new(),
        }
    }

    /// Add an event to the trace.
    pub fn push(&mut self, event: TraceEvent) {
        self.events.push(event);
    }

    /// Borrow the event list.
    #[must_use]
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// Borrow the source code.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Serialize to JSON string.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("trace serialization should not fail")
    }
}
