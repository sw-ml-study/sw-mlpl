//! Float presentations for a finite non-negative magnitude, matching
//! Python: fixed (`f`), exponent (`e`, two-digit signed exponent) and
//! general (`g`, fixed or exponent by magnitude, trailing zeros stripped);
//! plus MLPL's own display for `{}`.

/// `{:.pf}`: exactly `p` digits after the point.
pub fn fixed(a: f64, p: usize) -> String {
    format!("{a:.p$}")
}

/// `{:.pe}`: `d.ddd` + `e` + sign + at least two exponent digits.
pub fn exp(a: f64, p: usize, upper: bool) -> String {
    let s = format!("{a:.p$e}");
    let (mantissa, e) = s.split_once('e').unwrap_or((&s, "0"));
    let e: i32 = e.parse().unwrap_or(0);
    let sign = if e < 0 { '-' } else { '+' };
    let out = format!("{mantissa}e{sign}{:02}", e.abs());
    if upper { out.to_uppercase() } else { out }
}

/// `{:.pg}`: `p` significant digits; exponent form when the decimal
/// exponent is below -4 or at least `p`, fixed otherwise; trailing zeros
/// (and a bare point) removed.
pub fn general(a: f64, p: usize, upper: bool) -> String {
    let p = p.max(1);
    if a == 0.0 {
        return "0".into();
    }
    let strip = |m: &str| m.trim_end_matches('0').trim_end_matches('.').to_string();
    let e_form = exp(a, p - 1, upper);
    let (mantissa, e) = e_form.split_once(['e', 'E']).unwrap_or((&e_form, "+00"));
    let x: i32 = e.parse().unwrap_or(0);
    let p = i32::try_from(p).unwrap_or(i32::MAX);
    if x < -4 || x >= p {
        let marker = if upper { 'E' } else { 'e' };
        return format!("{}{marker}{e}", strip(mantissa));
    }
    strip(&fixed(a, usize::try_from(p - 1 - x).unwrap_or(0)))
}

/// MLPL's number display: integral values without a decimal part, others
/// in shortest round-trip form (`5`, `2.5`, `0.1`).
pub fn display(a: f64) -> String {
    if a.fract() == 0.0 && a.abs() < 1e15 {
        format!("{a:.0}")
    } else {
        format!("{a}")
    }
}
