//! Attention-heatmap payload. Populated when the evaluator
//! detects (or is told via `viz_hint`) that a tensor is
//! attention weights.
//!
//! Three layout variants cover the demos in the plan: rank-2
//! `[q, k]` for `attention.mlpl`-style raw matrices, rank-3
//! `[head, q, k]` for the per-head selector view, and rank-4
//! `[batch, head, q, k]` for tiny-LM-style real models.
//!
//! Token labels are deliberately `Vec<Token>` rather than
//! `Option<Vec<String>>` so the renderer can show integer
//! indices when no real vocab is available (`attention.mlpl`)
//! without a special-case None path.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionViz {
    pub query_tokens: Vec<Token>,
    pub key_tokens: Vec<Token>,
    /// Row-major weights. Indexing depends on `layout`.
    pub weights: Vec<f32>,
    pub layout: AttentionLayout,
    pub causal: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Token {
    Str(String),
    Index(usize),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "rank", rename_all = "snake_case")]
pub enum AttentionLayout {
    /// Rank-2 `[q, k]`.
    Qk { q: usize, k: usize },
    /// Rank-3 `[head, q, k]`. Saga C adds a head-selector
    /// dropdown to the renderer.
    HeadQk { head: usize, q: usize, k: usize },
    /// Rank-4 `[batch, head, q, k]`. Saga C adds a batch-slice
    /// dropdown alongside the head selector.
    BatchHeadQk {
        batch: usize,
        head: usize,
        q: usize,
        k: usize,
    },
}
