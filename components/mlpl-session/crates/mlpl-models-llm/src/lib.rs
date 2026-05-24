//! `llm_call(url, prompt, model)` -- thin runtime adapter
//! extracted from `crates/mlpl-eval/src/llm_dispatch.rs`.
//! Saga 33 step 016.
//!
//! Takes 3 pre-resolved strings (url, prompt, model), forwards
//! to `mlpl_runtime::call_ollama`, returns the reply string.
//! The caller (mlpl-eval) handles `Expr -> String` resolution
//! and the `String -> Value::Str` re-wrap at the boundary.

#[derive(Debug)]
pub enum LlmError {
    BadArity { expected: usize, got: usize },
    RuntimeMessage(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadArity { expected, got } => {
                write!(f, "llm_call expects {expected} arguments, got {got}")
            }
            Self::RuntimeMessage(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for LlmError {}

/// `llm_call(url, prompt, model)` -- the 3 strings must be
/// pre-resolved by the caller.
pub fn llm_call_inner(strs: &[String]) -> Result<String, LlmError> {
    let [url, prompt, model] = match strs {
        [a, b, c] => [a, b, c],
        _ => {
            return Err(LlmError::BadArity {
                expected: 3,
                got: strs.len(),
            });
        }
    };
    mlpl_runtime::call_ollama(url, prompt, model)
        .map_err(|e| LlmError::RuntimeMessage(format!("{e}")))
}
