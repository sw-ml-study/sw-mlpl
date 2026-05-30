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

/// `llm_call(url, prompt, model)` or `llm_call(url, prompt, model,
/// system)` -- the strings must be pre-resolved by the caller. The
/// optional 4th `system` arg populates Ollama's system/grounding
/// role (weak models follow it far better than an inlined preamble).
pub fn llm_call_inner(strs: &[String]) -> Result<String, LlmError> {
    let (url, prompt, model, system) = match strs {
        [a, b, c] => (a, b, c, ""),
        [a, b, c, d] => (a, b, c, d.as_str()),
        _ => {
            return Err(LlmError::BadArity {
                expected: 3,
                got: strs.len(),
            });
        }
    };
    mlpl_runtime::call_ollama_with_system(url, prompt, model, system)
        .map_err(|e| LlmError::RuntimeMessage(format!("{e}")))
}
