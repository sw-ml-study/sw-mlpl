//! Source location tracking.

/// A byte-offset range into source text.
///
/// `start` is inclusive, `end` is exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    /// Inclusive start byte offset.
    pub start: usize,
    /// Exclusive end byte offset.
    pub end: usize,
}

impl Span {
    /// Create a new span. Panics if `start > end`.
    #[must_use]
    pub fn new(start: usize, end: usize) -> Self {
        assert!(start <= end, "Span start ({start}) must be <= end ({end})");
        Self { start, end }
    }

    /// Number of bytes this span covers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether this span covers zero bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}
