//! Saga 76: free-function eval helpers extracted from lib.rs.

use mlpl_eval::Environment;
use mlpl_model_viz::model_to_viz_node;
use mlpl_web_viz_ir::VizNode;

pub struct EvalResult {
    pub display: String,
    pub values: Option<Vec<f64>>,
    pub shape: Vec<usize>,
    /// Saga BPE-1: populated when the evaluated value is a
    /// `Value::StrList` (produced by `decode_each(...)` or
    /// `[...]` string literals). Lets the viz layer attach
    /// per-token labels to attention sculptures.
    pub string_list: Option<Vec<String>>,
    /// Saga D: populated when the evaluated value is a
    /// `Value::Model`. Carries a pre-built `VizNode` with a
    /// composite-Sankey payload describing the chain's flow
    /// (one node per layer, one edge per layer-to-layer
    /// connection). The JS dispatch site hands it to the
    /// `composite` renderer.
    pub viz_node: Option<VizNode>,
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
        Ok(value) => {
            let mut result = value_to_result(value);
            // Saga D fix: `mdl = attention(...)` and other
            // assignments-to-Model bind the Model into env.models
            // and return a scalar placeholder, so the
            // value_to_result match arm never sees a Value::Model
            // for assigned models. Look up the assigned variable
            // in env.models when the eval path produced no viz
            // and the line looks like a plain assignment. Cheap
            // -- it's a HashMap hit on a name the user just typed.
            if result.viz_node.is_none()
                && let Some(name) = extract_assigned_name(trimmed)
                && let Some(spec) = env.get_model(&name)
            {
                result.viz_node = Some(model_to_viz_node(spec));
            }
            result
        }
        Err(e) => text_result(format!("error: {e}")),
    }
}

/// Extract the LHS identifier of `<name> = <expr>`. Returns
/// `None` for non-assignment lines (no `=`) or anything where
/// the LHS contains non-identifier characters. Used by the
/// saga D viz lookup to find a Model that was just bound to
/// `name`.
fn extract_assigned_name(line: &str) -> Option<String> {
    let eq_pos = line.find('=')?;
    let lhs = line[..eq_pos].trim();
    if lhs.is_empty() {
        return None;
    }
    if !lhs.chars().all(|c| c.is_alphanumeric() || c == '_')
        || !lhs
            .chars()
            .next()
            .is_some_and(|c| c.is_alphabetic() || c == '_')
    {
        return None;
    }
    Some(lhs.to_string())
}

fn empty_result() -> EvalResult {
    EvalResult {
        display: String::new(),
        values: None,
        shape: vec![],
        string_list: None,
        viz_node: None,
    }
}

fn text_result(display: String) -> EvalResult {
    EvalResult {
        display,
        values: None,
        shape: vec![],
        string_list: None,
        viz_node: None,
    }
}

fn value_to_result(value: mlpl_eval::Value) -> EvalResult {
    match value {
        mlpl_eval::Value::Array(ref arr) => EvalResult {
            display: format!("{}", mlpl_eval::Value::Array(arr.clone())),
            values: Some(arr.data().to_vec()),
            shape: arr.shape().dims().to_vec(),
            string_list: None,
            viz_node: None,
        },
        mlpl_eval::Value::StrList { ref items } => EvalResult {
            display: format!(
                "{}",
                mlpl_eval::Value::StrList {
                    items: items.clone()
                }
            ),
            values: None,
            shape: vec![items.len()],
            string_list: Some(items.clone()),
            viz_node: None,
        },
        mlpl_eval::Value::Model(ref spec) => EvalResult {
            display: format!("{}", mlpl_eval::Value::Model(spec.clone())),
            values: None,
            shape: vec![],
            string_list: None,
            viz_node: Some(model_to_viz_node(spec)),
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
