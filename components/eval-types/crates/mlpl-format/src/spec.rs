//! Parse one replacement field's format spec -- the part after `:` in
//! `{:>8.2f}` -- following Python's grammar
//! `[[fill]align][sign][#][0][width][,][.precision][type]`.

/// A parsed format spec.
#[derive(Clone, Debug)]
pub struct Spec {
    /// Padding character (default space; `0` with the zero flag).
    pub fill: char,
    /// `<` `>` `^` `=` or none (numbers right, text left).
    pub align: Option<char>,
    /// `+` (always), ` ` (space for non-negative) or `-` (default).
    pub sign: char,
    /// `#`: prefix `0x` / `0b` / `0o`.
    pub alt: bool,
    /// Minimum field width.
    pub width: usize,
    /// `,`: thousands separators.
    pub comma: bool,
    /// Digits after the point (or significant digits for `g`).
    pub precision: Option<usize>,
    /// Presentation type (`d x X b o f F e E g G % s`) or none.
    pub ty: Option<char>,
}

/// The empty spec (`{}`): every option at its default.
const BLANK: Spec = Spec {
    fill: ' ',
    align: None,
    sign: '-',
    alt: false,
    width: 0,
    comma: false,
    precision: None,
    ty: None,
};

/// Parse `spec`; the error names what was not understood.
pub fn parse_spec(spec: &str) -> Result<Spec, String> {
    let c: Vec<char> = spec.chars().collect();
    let is_align = |ch: Option<&char>| ch.is_some_and(|a| "<>^=".contains(*a));
    let mut out = BLANK;
    let mut i = 0;
    if is_align(c.get(1)) {
        (out.fill, out.align, i) = (c[0], Some(c[1]), 2);
    } else if is_align(c.first()) {
        (out.align, i) = (Some(c[0]), 1);
    }
    if let Some(&s) = c.get(i).filter(|s| "+- ".contains(**s)) {
        (out.sign, i) = (s, i + 1);
    }
    if c.get(i) == Some(&'#') {
        (out.alt, i) = (true, i + 1);
    }
    if c.get(i) == Some(&'0') && out.align.is_none() {
        (out.fill, out.align) = ('0', Some('='));
    }
    parse_tail(&c, i, out)
}

/// Width, `,`, `.precision` and the type -- then nothing may remain.
fn parse_tail(c: &[char], mut i: usize, mut out: Spec) -> Result<Spec, String> {
    out.width = digits(c, &mut i).unwrap_or(0);
    if c.get(i) == Some(&',') {
        (out.comma, i) = (true, i + 1);
    }
    if c.get(i) == Some(&'.') {
        i += 1;
        out.precision = Some(digits(c, &mut i).ok_or("a '.' must be followed by a precision")?);
    }
    match c.get(i..) {
        Some([]) | None => Ok(out),
        Some([t]) if "dxXbofFeEgG%s".contains(*t) => Ok(Spec {
            ty: Some(*t),
            ..out
        }),
        Some([t]) => Err(format!("unknown format type '{t}'")),
        Some(rest) => Err(format!(
            "unexpected '{}' in the format spec",
            rest.iter().collect::<String>()
        )),
    }
}

/// The decimal number starting at `c[*i]`, advancing `*i` past it.
fn digits(c: &[char], i: &mut usize) -> Option<usize> {
    let start = *i;
    while c.get(*i).is_some_and(char::is_ascii_digit) {
        *i += 1;
    }
    c[start..*i].iter().collect::<String>().parse().ok()
}
