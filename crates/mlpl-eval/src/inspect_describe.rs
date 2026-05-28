//! `:describe <name>` rendering split by value kind.
//!
//! `format_describe` is the thin dispatcher: it picks the right
//! describer for tokenizers / models / arrays / strings, with the
//! built-in registry as the final fallback. The original 46-LOC
//! function fanned all five branches inline (saga 33 step 024
//! split them into per-kind helpers so each stays under the
//! 25-LOC function-LOC budget).

use mlpl_array::DenseArray;
use mlpl_eval_core::inspect_groups::BUILTIN_GROUPS;
use mlpl_eval_core::model::ModelSpec;

use crate::env::Environment;

pub(crate) fn format_describe(env: &Environment, name: &str) -> String {
    if let Some(desc) = env.describe_fn(name) {
        return desc;
    }
    if let Some(tok) = env.tokenizers.get(name) {
        return format!("{name} -- tokenizer\n  {}", tok.describe());
    }
    if let Some(spec) = env.models.get(name) {
        return describe_model(env, name, spec);
    }
    if let Some(arr) = env.vars.get(name) {
        return describe_array(env, name, arr);
    }
    if let Some(s) = env.get_string(name) {
        let body = s
            .lines()
            .map(|l| format!("  {l}"))
            .collect::<Vec<_>>()
            .join("\n");
        return format!("{name} -- string ({} chars)\n{body}", s.len());
    }
    for (_, entries) in BUILTIN_GROUPS {
        if let Some(doc) = entries.iter().find(|(n, _, _)| *n == name) {
            return format!("{} -- built-in\n  {}\n  {}", doc.0, doc.1, doc.2);
        }
    }
    format!("'{name}' is not a bound variable, model, or built-in.")
}

fn describe_model(env: &Environment, name: &str, spec: &ModelSpec) -> String {
    let mut out = format!(
        "{name} -- model\n  shape: {}\n",
        crate::inspect_render::render_spec(spec)
    );
    let ps = spec.params();
    if ps.is_empty() {
        out.push_str("  params: (none)");
        return out;
    }
    out.push_str("  params:\n");
    for p in ps {
        if let Some(arr) = env.vars.get(&p) {
            out.push_str(&format!(
                "    {p}: {}\n",
                crate::inspect_render::format_shape(arr)
            ));
        }
    }
    out.truncate(out.trim_end().len());
    out
}

fn describe_array(env: &Environment, name: &str, arr: &DenseArray) -> String {
    let shape = crate::inspect_render::format_shape(arr);
    let param_marker = if env.params.contains(name) {
        " (trainable param)"
    } else {
        ""
    };
    let preview = array_preview(arr.data());
    let header = match env.get_tag(name) {
        Some(t) => format!("{name} -- {}", crate::tag_render::header_line(t)),
        None => format!("{name} -- array"),
    };
    let mut out = format!("{header}\n  shape: {shape}{param_marker}\n  values: {preview}");
    if let Some(t) = env.get_tag(name) {
        for line in crate::tag_render::body_lines(t, Some(arr)) {
            out.push_str(&format!("\n  {line}"));
        }
    }
    out
}

fn array_preview(data: &[f64]) -> String {
    if data.is_empty() {
        return "(empty)".into();
    }
    let take = 8.min(data.len());
    let head: Vec<String> = data[..take].iter().map(|v| format!("{v:.4}")).collect();
    if data.len() > take {
        format!("{} ... ({} total)", head.join(" "), data.len())
    } else {
        head.join(" ")
    }
}
