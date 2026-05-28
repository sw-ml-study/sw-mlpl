//! RNN built-in functions extracted from mlpl-runtime.

mod lstm;
mod rnn;

pub const NAMES: &[&str] = &["rnn_cell", "lstm_cell"];

pub use rnn::try_call;
