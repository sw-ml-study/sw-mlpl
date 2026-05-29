//! Prefix extraction + candidate matching + completion application.
//! Pure string helpers; no Yew or wasm-bindgen dependency.

/// Characters that count as part of a completable token.
/// Alphanumeric + underscore + leading colon (for `:cmds`).
fn is_token_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == ':'
}

/// Find the token prefix ending at `cursor` in `input`.
/// Returns `(prefix_start, prefix)` -- empty `prefix` means
/// the cursor is on whitespace and there's nothing to
/// complete. `cursor` is clamped to `input.len()`.
pub fn extract_prefix(input: &str, cursor: usize) -> (usize, &str) {
    let cursor = cursor.min(input.len());
    let mut start = cursor;
    for (i, c) in input[..cursor].char_indices().rev() {
        if is_token_char(c) {
            start = i;
        } else {
            break;
        }
    }
    (start, &input[start..cursor])
}

/// Filter `candidates` to those starting with `prefix`.
/// Returns an owned sorted-deduped Vec to keep callers
/// simple. Empty prefix returns empty (nothing to complete).
pub fn match_candidates(prefix: &str, candidates: &[&str]) -> Vec<String> {
    if prefix.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<String> = candidates
        .iter()
        .filter(|c| c.starts_with(prefix) && **c != prefix)
        .map(|c| (*c).to_string())
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Apply `completion` at `cursor` in `input`, replacing the
/// token prefix at the cursor. Returns `(new_input,
/// new_cursor)` where the cursor lands at the end of the
/// inserted completion.
pub fn apply_completion(input: &str, cursor: usize, completion: &str) -> (String, usize) {
    let (start, _prefix) = extract_prefix(input, cursor);
    let cursor = cursor.min(input.len());
    let mut out = String::with_capacity(input.len() + completion.len());
    out.push_str(&input[..start]);
    out.push_str(completion);
    out.push_str(&input[cursor..]);
    let new_cursor = start + completion.len();
    (out, new_cursor)
}
