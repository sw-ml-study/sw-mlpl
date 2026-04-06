//! Execution tracing for MLPL.
//!
//! Records evaluation steps for replay, visualization, and debugging.

mod event;
mod trace;
mod value;

pub use event::TraceEvent;
pub use trace::Trace;
pub use value::TraceValue;
