//! Render one argument against a parsed spec, then pad it to the field
//! width. Text aligns left by default, numbers right; `=` (and the `0`
//! flag) pads between the sign / radix prefix and the digits.

use crate::spec::Spec;
use crate::template::FmtArg;

/// The formatted, padded text of `arg`; errors name the field spec `raw`
/// and the argument index.
pub fn render(arg: &FmtArg, spec: &Spec, raw: &str, idx: usize) -> Result<String, String> {
    let (body, default_align) = match arg {
        FmtArg::Text(_) if spec.ty.is_some_and(|t| t != 's') => {
            return Err(format!(
                "format: {{:{raw}}} needs a number, got a string (argument {idx})"
            ));
        }
        FmtArg::Text(t) => (t.clone(), '<'),
        FmtArg::Num(v) => {
            let body = crate::number::format_number(*v, spec)
                .map_err(|m| format!("format: {{:{raw}}} {m} (argument {idx})"))?;
            (body, '>')
        }
    };
    Ok(pad(&body, spec, default_align))
}

/// Pad `body` to `spec.width` with `spec.fill`, honoring the alignment.
fn pad(body: &str, spec: &Spec, default_align: char) -> String {
    let n = spec.width.saturating_sub(body.chars().count());
    let fill = |k: usize| spec.fill.to_string().repeat(k);
    match spec.align.unwrap_or(default_align) {
        '<' => format!("{body}{}", fill(n)),
        '^' => format!("{}{body}{}", fill(n / 2), fill(n - n / 2)),
        '=' => {
            let (prefix, digits) = body.split_at(prefix_len(body));
            format!("{prefix}{}{digits}", fill(n))
        }
        _ => format!("{}{body}", fill(n)),
    }
}

/// Length of a leading sign and radix prefix (`-`, `+`, ` `, `0x`, ...).
fn prefix_len(body: &str) -> usize {
    let sign = usize::from(body.starts_with(['-', '+', ' ']));
    let rest = &body[sign..];
    let radix = ["0x", "0X", "0b", "0o"].iter().any(|p| rest.starts_with(p));
    sign + if radix { 2 } else { 0 }
}
