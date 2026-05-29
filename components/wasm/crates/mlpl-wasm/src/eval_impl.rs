//! Saga 76: free-function eval helpers extracted from lib.rs.

use mlpl_eval::Environment;

pub struct EvalResult {
    pub display: String,
    pub values: Option<Vec<f64>>,
    pub shape: Vec<usize>,
    /// Saga BPE-1: populated when the evaluated value is a
    /// `Value::StrList` (produced by `decode_each(...)` or
    /// `[...]` string literals). Lets the viz layer attach
    /// per-token labels to attention sculptures.
    pub string_list: Option<Vec<String>>,
}

pub(crate) fn eval_input_with_values(input: &str, env: &mut Environment) -> EvalResult {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return empty_result();
    }
    if let Some(out) = mlpl_eval::inspect(env, trimmed) {
        return text_result(out);
    }
    let tokens = match mlpl_parser::lex(trimmed) {
        Ok(t) => t,
        Err(e) => return text_result(format!("error: {e}")),
    };
    let stmts = match mlpl_parser::parse(&tokens) {
        Ok(s) if s.is_empty() => return empty_result(),
        Ok(s) => s,
        Err(e) => return text_result(format!("error: {e}")),
    };
    match mlpl_eval::eval_program_value(&stmts, env) {
        Ok(value) => value_to_result(value),
        Err(e) => text_result(format!("error: {e}")),
    }
}

fn empty_result() -> EvalResult {
    EvalResult {
        display: String::new(),
        values: None,
        shape: vec![],
        string_list: None,
    }
}

fn text_result(display: String) -> EvalResult {
    EvalResult {
        display,
        values: None,
        shape: vec![],
        string_list: None,
    }
}

fn value_to_result(value: mlpl_eval::Value) -> EvalResult {
    match value {
        mlpl_eval::Value::Array(ref arr) => EvalResult {
            display: format!("{}", mlpl_eval::Value::Array(arr.clone())),
            values: Some(arr.data().to_vec()),
            shape: arr.shape().dims().to_vec(),
            string_list: None,
        },
        mlpl_eval::Value::StrList { ref items } => EvalResult {
            display: format!("{}", mlpl_eval::Value::StrList { items: items.clone() }),
            values: None,
            shape: vec![items.len()],
            string_list: Some(items.clone()),
        },
        other => text_result(format!("{other}")),
    }
}

pub(crate) fn eval_input(input: &str, env: &mut Environment) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Introspection commands (`:vars`, `:models`, `:fns`, `:wsid`,
    // `:describe <name>`) short-circuit evaluation so they work
    // identically in the terminal and web REPLs.
    if let Some(out) = mlpl_eval::inspect(env, trimmed) {
        return out;
    }

    let tokens = match mlpl_parser::lex(trimmed) {
        Ok(t) => t,
        Err(e) => return format!("error: {e}"),
    };

    let stmts = match mlpl_parser::parse(&tokens) {
        Ok(s) if s.is_empty() => return String::new(),
        Ok(s) => s,
        Err(e) => return format!("error: {e}"),
    };

    match mlpl_eval::eval_program_value(&stmts, env) {
        Ok(value) => format!("{value}"),
        Err(e) => format!("error: {e}"),
    }
}
