//! Walk a template: literal text, `{{` / `}}` escapes, and replacement
//! fields `{}` (automatic numbering) / `{N}` (manual), each with an
//! optional `:spec`. Mixing automatic and manual numbering is an error, as
//! in Python.

/// One `format` argument: a scalar number or text (a string, or the
/// display text of a non-scalar value).
#[derive(Clone, Debug, PartialEq)]
pub enum FmtArg {
    /// A scalar number.
    Num(f64),
    /// A string (formats only with `{}` / `s` / alignment specs).
    Text(String),
}

/// Render `template` with `args`.
///
/// # Errors
///
/// A message naming the field, the argument and what went wrong: an
/// unmatched brace, mixed automatic / manual numbering, a missing
/// argument, an unknown spec, or a type mismatch (e.g. `{:d}` on 2.5).
pub fn format(template: &str, args: &[FmtArg]) -> Result<String, String> {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    let mut numbering: Option<bool> = None; // Some(true) = automatic
    let mut field = 0;
    while let Some(c) = chars.next() {
        match c {
            '{' | '}' if chars.peek() == Some(&c) => {
                chars.next();
                out.push(c);
            }
            '{' => {
                let body = read_field(&mut chars)?;
                let idx = field_index(&body, field, &mut numbering)?;
                out.push_str(&render_field(&body, idx, field, args)?);
                field += 1;
            }
            '}' => {
                return Err("format: single '}' in the template (write '}}' for a brace)".into());
            }
            c => out.push(c),
        }
    }
    Ok(out)
}

/// The text of a replacement field up to its closing `}` (consumed).
fn read_field(chars: &mut impl Iterator<Item = char>) -> Result<String, String> {
    let mut body = String::new();
    for ch in chars {
        if ch == '}' {
            return Ok(body);
        }
        body.push(ch);
    }
    Err("format: unmatched '{' in the template (write '{{' for a brace)".into())
}

/// The argument index a field refers to, enforcing one numbering style.
fn field_index(body: &str, field: usize, numbering: &mut Option<bool>) -> Result<usize, String> {
    let name = body.split(':').next().unwrap_or("");
    let auto = name.is_empty();
    if numbering.is_some_and(|a| a != auto) {
        return Err("format: cannot switch between automatic field numbering ({}) and manual field specification ({0})".into());
    }
    *numbering = Some(auto);
    if auto {
        return Ok(field);
    }
    name.parse().map_err(|_| {
        format!("format: field {field} has index '{name}'; use a number such as {{0}}")
    })
}

/// Render one field `{index:spec}` against its argument.
fn render_field(body: &str, idx: usize, field: usize, args: &[FmtArg]) -> Result<String, String> {
    let arg = args.get(idx).ok_or_else(|| {
        format!(
            "format: field {field} needs argument {idx} but only {} given",
            args.len()
        )
    })?;
    let raw = body.split_once(':').map_or("", |(_, s)| s);
    let spec = crate::spec::parse_spec(raw).map_err(|e| format!("format: {{:{raw}}}: {e}"))?;
    crate::render::render(arg, &spec, raw, idx)
}
