//! Binary fixture parsing for the pets-tiny dataset: header preamble
//! checks and section slicing. Split from `lib.rs` (connect-telemetry
//! step 006 ratchet) per the parse.rs file-naming convention.

use mlpl_eval_types::EvalError;

use crate::{MAGIC, VERSION, corrupt};

pub(crate) struct Header {
    pub(crate) n: usize,
    pub(crate) c: usize,
    pub(crate) h: usize,
    pub(crate) w: usize,
}

pub(crate) fn parse_header(bytes: &[u8]) -> Result<(Header, &[u8]), EvalError> {
    check_preamble(bytes)?;
    let read_u32 = |o: usize| {
        u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize
    };
    let n = read_u32(12);
    let c = read_u32(16);
    let h = read_u32(20);
    let w = read_u32(24);
    Ok((Header { n, c, h, w }, &bytes[28..]))
}

/// Emptiness / size / magic / version checks ahead of header decode.
fn check_preamble(bytes: &[u8]) -> Result<(), EvalError> {
    if bytes.is_empty() {
        return Err(EvalError::Unsupported(
            "load_preloaded(\"pets_tiny\"): fixture is empty -- run \
             scripts/build-pets-tiny.rs against the gitignored \
             data/oxford-iiit-pet/ checkout to populate it"
                .into(),
        ));
    }
    if bytes.len() < 8 + 4 + 16 {
        return Err(corrupt("file shorter than header"));
    }
    if &bytes[..8] != MAGIC {
        return Err(corrupt("bad magic bytes"));
    }
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if version != VERSION {
        return Err(EvalError::Unsupported(format!(
            "load_preloaded(\"pets_tiny\"): version {version} not supported (this build expects {VERSION})"
        )));
    }
    Ok(())
}

pub(crate) struct Sections<'a> {
    pub(crate) y_bytes: &'a [u8],
    pub(crate) names_bytes: &'a [u8],
    pub(crate) pixels: &'a [u8],
}

pub(crate) fn parse_sections<'a>(body: &'a [u8], h: &Header) -> Result<Sections<'a>, EvalError> {
    let Header { n, c, h: ph, w } = *h;
    let mut cur = 0;
    let y_bytes = take(body, cur, n, "Y labels")?;
    cur += n;
    let len_bytes = take(body, cur, 4, "name-table length")?;
    let names_len =
        u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
    cur += 4;
    let names_bytes = take(body, cur, names_len, "name-table")?;
    cur += names_len;
    let pixels = take(body, cur, n * c * ph * w, "pixel data")?;
    Ok(Sections {
        y_bytes,
        names_bytes,
        pixels,
    })
}

/// Bounds-checked slice `body[cur..cur + len]`, or a corrupt error
/// naming the truncated section.
fn take<'a>(body: &'a [u8], cur: usize, len: usize, what: &str) -> Result<&'a [u8], EvalError> {
    if body.len() < cur + len {
        return Err(corrupt(&format!("{what} truncated")));
    }
    Ok(&body[cur..cur + len])
}
