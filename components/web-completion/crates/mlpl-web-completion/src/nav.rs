//! Popup navigation helpers: wrap-around index increment /
//! decrement and the ArrowRight accept predicate.

/// Wrap-around index increment for popup navigation.
#[must_use]
pub fn next_index(cur: usize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    (cur + 1) % len
}

/// Wrap-around index decrement for popup navigation.
#[must_use]
pub fn prev_index(cur: usize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    if cur == 0 { len - 1 } else { cur - 1 }
}

/// ArrowRight should accept the highlighted candidate only
/// when the popup is open AND the cursor is at the end of
/// the input (otherwise ArrowRight is a normal cursor move).
#[must_use]
pub fn should_accept_right(popup_open: bool, cursor: usize, input_len: usize) -> bool {
    popup_open && cursor >= input_len
}
