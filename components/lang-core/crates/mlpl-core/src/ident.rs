//! Validated identifier type.

/// A validated MLPL identifier.
///
/// Must start with an ASCII letter or underscore, followed by
/// ASCII letters, digits, or underscores. Never empty.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Identifier(String);

impl Identifier {
    /// Create a new identifier from a string, validating it.
    ///
    /// Returns `None` if the string is not a valid identifier.
    pub fn new(s: &str) -> Option<Self> {
        if is_valid_ident(s) {
            Some(Self(s.to_owned()))
        } else {
            None
        }
    }

    /// Return the identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Check whether a string is a valid MLPL identifier.
///
/// Rules (from syntax-core-v1):
/// - Must not be empty
/// - First character: ASCII letter or underscore
/// - Subsequent characters: ASCII letters, digits, or underscores
fn is_valid_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}
