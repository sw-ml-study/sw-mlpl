//! Parse an equation line into positioned ATOMS. Most of the line is
//! ordinary text runs (which keep inline `_x`/`^x` sub/superscripts); a
//! big operator (sum, product, integral) followed by `_{...}`/`^{...}`
//! becomes a `BigOp` whose limits render STACKED above and below it, the
//! display convention a paper uses.

use crate::spans::take_script;

/// One left-to-right unit of an equation line.
pub(crate) enum Atom {
    /// A plain run; may still contain inline `_x`/`^x` handled by `spans`.
    Text(String),
    /// A big operator with optional lower (`sub`) and upper (`sup`) limits.
    BigOp {
        op: String,
        sub: Option<String>,
        sup: Option<String>,
    },
}

/// Split `line` into atoms: each big operator (with the limits that
/// immediately follow it) is its own atom; everything else coalesces
/// into text runs.
pub(crate) fn atoms(line: &str) -> Vec<Atom> {
    let mut out = Vec::new();
    let mut text = String::new();
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if !is_big_op(c) {
            text.push(c);
            continue;
        }
        if !text.is_empty() {
            out.push(Atom::Text(std::mem::take(&mut text)));
        }
        let (sub, sup) = read_limits(&mut chars);
        out.push(Atom::BigOp {
            op: c.to_string(),
            sub,
            sup,
        });
    }
    if !text.is_empty() {
        out.push(Atom::Text(text));
    }
    out
}

/// The operators that take stacked limits: sum, product, integral, and
/// big union / intersection.
fn is_big_op(c: char) -> bool {
    matches!(
        c,
        '\u{2211}' | '\u{220F}' | '\u{222B}' | '\u{22C3}' | '\u{22C2}'
    )
}

/// Read up to two limits (`_`/`^`, either order) directly following a
/// big operator, returning `(lower, upper)`.
fn read_limits(
    chars: &mut std::iter::Peekable<std::str::Chars>,
) -> (Option<String>, Option<String>) {
    let (mut sub, mut sup) = (None, None);
    for _ in 0..2 {
        match chars.peek() {
            Some('_') => {
                chars.next();
                sub = Some(take_script(chars));
            }
            Some('^') => {
                chars.next();
                sup = Some(take_script(chars));
            }
            _ => break,
        }
    }
    (sub, sup)
}
