//! Connect-mode terminal rendering: workspace-snapshot formatting for
//! the `:vars` / `:models` / ... slash commands and the mid-train
//! cancellation report. Split from `connect_repl.rs` (connect-telemetry
//! step 004) to bring that module back inside the function-count budget.

use crate::connect::InspectResponse;

pub(crate) fn render_cancellation(step: usize, partial_losses: &[f64]) {
    eprintln!(
        "  cancelled at step {step} ({} partial loss{} captured; see :vars last_losses)",
        partial_losses.len(),
        if partial_losses.len() == 1 { "" } else { "es" }
    );
    if !partial_losses.is_empty() {
        let preview: Vec<String> = partial_losses
            .iter()
            .take(5)
            .map(|v| format!("{v:.4}"))
            .collect();
        let ellipsis = if partial_losses.len() > 5 {
            ", ..."
        } else {
            ""
        };
        eprintln!("  losses: [{}{ellipsis}]", preview.join(", "));
    }
}

pub(crate) fn format_inspect(command: &str, snap: &InspectResponse) -> String {
    let mut out = String::new();
    let render_names = |out: &mut String, label: &str, names: &[String]| {
        if names.is_empty() {
            out.push_str(&format!("(no {label})"));
        } else {
            for n in names {
                out.push_str(&format!("  {n}\n"));
            }
            out.truncate(out.trim_end().len());
        }
    };
    match command {
        ":vars" => out.push_str(&format_vars(snap)),
        ":models" => render_names(&mut out, "models", &snap.models),
        ":tokenizers" => render_names(&mut out, "tokenizers", &snap.tokenizers),
        ":experiments" => render_names(&mut out, "experiments", &snap.experiments),
        ":wsid" => {
            out.push_str(&format_wsid(snap));
        }
        _ => unreachable!("dispatch_slash filters before format_inspect"),
    }
    out
}

/// The `:vars` listing: name, shape, `[param]` tag, truncation note.
fn format_vars(snap: &InspectResponse) -> String {
    let mut out = String::new();
    if snap.vars.is_empty() {
        out.push_str("(no variables)");
        return out;
    }
    for v in &snap.vars {
        let tag = if v.is_param { " [param]" } else { "" };
        let dims: Vec<String> = v.shape.iter().map(|d| d.to_string()).collect();
        out.push_str(&format!("  {}: [{}]{tag}\n", v.name, dims.join(", ")));
    }
    if snap.more > 0 {
        out.push_str(&format!("  ... ({} more)\n", snap.more));
    }
    out.truncate(out.trim_end().len());
    out
}

/// The `:wsid` workspace summary counts.
fn format_wsid(snap: &InspectResponse) -> String {
    format!(
        "workspace (remote):\n  variables: {}\n  models:    {}\n  tokenizers: {}\n  experiments: {}",
        snap.vars.len() + snap.more,
        snap.models.len(),
        snap.tokenizers.len(),
        snap.experiments.len()
    )
}
