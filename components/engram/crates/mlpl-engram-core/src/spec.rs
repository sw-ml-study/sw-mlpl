//! `EngramSpec` -- the backend-neutral description of one Engram
//! module -- and its validation. Contains no tensor types: every
//! backend (CPU reference, MLX, later CUDA) is configured from
//! this one struct, and the REPL's `:describe` introspection reads
//! the derived accounting in `accounting.rs`.

/// Description of one Engram memory module (per injected layer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngramSpec {
    /// Width of the residual stream the module injects into.
    pub hidden_size: usize,
    /// N-gram window sizes, e.g. `[2, 3]` for bigram + trigram.
    pub ngram_orders: Vec<usize>,
    /// Independent hash heads per n-gram order.
    pub heads_per_ngram: usize,
    /// Table slots per head (per order).
    pub slots_per_head: usize,
    /// Dimensions stored per slot (LOW, deliberately -- the doc's
    /// core correction over full-hidden-width tables).
    pub head_dim: usize,
    /// Seed for the deterministic hash multipliers.
    pub seed: u64,
}

/// Everything that can be wrong with an [`EngramSpec`] (or with
/// inputs fed to the hash reference).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngramError {
    EmptyNgramOrders,
    NgramOrderTooSmall { order: usize },
    ZeroField { field: &'static str },
    Overflow,
    TokenIdTooLarge { id: u64 },
}

impl std::fmt::Display for EngramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyNgramOrders => write!(f, "engram: ngram_orders must not be empty"),
            Self::NgramOrderTooSmall { order } => {
                write!(f, "engram: ngram order {order} too small (minimum 2)")
            }
            Self::ZeroField { field } => write!(f, "engram: {field} must be non-zero"),
            Self::Overflow => write!(f, "engram: parameter count overflows"),
            Self::TokenIdTooLarge { id } => write!(
                f,
                "engram: token id {id} exceeds the exact-arithmetic bound \
                 ({} = 2^21 - 1)",
                crate::MAX_TOKEN_ID
            ),
        }
    }
}

impl std::error::Error for EngramError {}

impl EngramSpec {
    /// Check the spec's structural invariants.
    ///
    /// # Errors
    /// The first violated invariant, as an [`EngramError`].
    pub fn validate(&self) -> Result<(), EngramError> {
        if self.ngram_orders.is_empty() {
            return Err(EngramError::EmptyNgramOrders);
        }
        if let Some(&order) = self.ngram_orders.iter().find(|&&o| o < 2) {
            return Err(EngramError::NgramOrderTooSmall { order });
        }
        for (field, value) in [
            ("heads_per_ngram", self.heads_per_ngram),
            ("slots_per_head", self.slots_per_head),
            ("head_dim", self.head_dim),
            ("hidden_size", self.hidden_size),
        ] {
            if value == 0 {
                return Err(EngramError::ZeroField { field });
            }
        }
        self.parameter_count().map(|_| ())
    }
}
