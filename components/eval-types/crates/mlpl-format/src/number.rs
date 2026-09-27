//! Numbers against a spec: sign, integer / radix types, thousands
//! grouping; the float presentations live in `float`.

use crate::float::{display, exp, fixed, general};
use crate::spec::Spec;

/// The signed digits of `v` for `spec` (no padding). `Err` explains a
/// type mismatch, e.g. a non-integer for `{:d}`.
pub fn format_number(v: f64, spec: &Spec) -> Result<String, String> {
    let integral_ty = spec.ty.is_some_and(|t| "dxXbo".contains(t));
    if integral_ty && v.is_finite() && v.fract() != 0.0 {
        return Err(format!("needs an integer, got {}", display(v)));
    }
    let upper = spec.ty.is_some_and(|t| t.is_ascii_uppercase());
    let digits = if v.is_finite() {
        magnitude(v.abs(), spec, upper)
    } else {
        let t = if v.is_nan() { "nan" } else { "inf" };
        if upper {
            t.to_uppercase()
        } else {
            t.to_string()
        }
    };
    let sign = match (v < 0.0, spec.sign) {
        (true, _) => "-",
        (false, '+') => "+",
        (false, ' ') => " ",
        _ => "",
    };
    Ok(format!("{sign}{digits}"))
}

/// The unsigned digits of a finite non-negative `a` for `spec`'s type.
fn magnitude(a: f64, spec: &Spec, upper: bool) -> String {
    let p = spec.precision;
    let comma = spec.comma;
    match spec.ty {
        None if p.is_none() => group(&display(a), comma),
        None | Some('g' | 'G') => general(a, p.unwrap_or(6), upper),
        Some('d') => group(&display(a), comma),
        Some('x' | 'X' | 'b' | 'o') => radix(a, spec),
        Some('f' | 'F') => group(&fixed(a, p.unwrap_or(6)), comma),
        Some('%') => format!("{}%", fixed(a * 100.0, p.unwrap_or(6))),
        Some('e' | 'E') => exp(a, p.unwrap_or(6), upper),
        Some(_) => display(a),
    }
}

/// Hex / binary / octal digits of an integral `a`, with the `#` prefix.
fn radix(a: f64, spec: &Spec) -> String {
    // `a` is integral and non-negative (checked by format_number); go
    // through its exact decimal text rather than a lossy cast.
    let n: u128 = format!("{a:.0}").parse().unwrap_or(0);
    let (digits, prefix) = match spec.ty {
        Some('x') => (format!("{n:x}"), "0x"),
        Some('X') => (format!("{n:X}"), "0X"),
        Some('b') => (format!("{n:b}"), "0b"),
        _ => (format!("{n:o}"), "0o"),
    };
    if spec.alt {
        format!("{prefix}{digits}")
    } else {
        digits
    }
}

/// Insert `,` every three digits of the integer part when `comma`.
fn group(digits: &str, comma: bool) -> String {
    if !comma {
        return digits.to_string();
    }
    let (int, frac) = digits.split_at(digits.find('.').unwrap_or(digits.len()));
    let mut out = String::new();
    for (i, ch) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out + frac
}
