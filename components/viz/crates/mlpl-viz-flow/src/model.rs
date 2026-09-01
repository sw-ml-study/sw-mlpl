//! Typed model for the dataflow renderer: a validated graph and its
//! laid-out positions. Node labels + directed edges (optional per-edge
//! labels), plus the Phase 2 channels -- node groups, per-node and
//! per-edge highlight, and per-edge width (docs/dataflow-renderer-design.md).

/// A validated dataflow graph. Node ids are indices into `labels`;
/// every edge endpoint is in range, and each optional channel is either
/// empty (unused) or exactly one-per-node / one-per-edge (guaranteed by
/// `build`).
pub struct Graph {
    /// One box label per node; index = node id.
    pub labels: Vec<String>,
    /// Directed edges as `(from_id, to_id)` pairs.
    pub edges: Vec<(usize, usize)>,
    /// Per-edge label, or empty for none.
    pub edge_labels: Vec<String>,
    /// Group id per node (same id -> banded together), or empty.
    pub groups: Vec<usize>,
    /// Per-node highlight flag, or empty.
    pub node_highlight: Vec<bool>,
    /// Per-edge stroke width, or empty for the default.
    pub edge_widths: Vec<f64>,
    /// Per-edge highlight flag, or empty.
    pub edge_highlight: Vec<bool>,
    /// Interpret `edge_widths` on a log scale (honest for extreme
    /// ratios) instead of clamped-linear.
    pub width_log: bool,
}

/// A graph with a pixel position assigned to every node, plus the
/// canvas size the render step draws into.
pub struct Positioned {
    /// The source graph.
    pub graph: Graph,
    /// Top-left `(x, y)` of each node box, indexed by node id.
    pub pos: Vec<(i32, i32)>,
    /// Per-edge back-edge flag (a recurrence): drawn dashed through the
    /// bottom lane and excluded from layering.
    pub back: Vec<bool>,
    /// Canvas width / height in pixels.
    pub width: i32,
    /// Canvas height.
    pub height: i32,
}

/// Node box width in pixels.
pub const NODE_W: i32 = 128;
/// Node box height in pixels.
pub const NODE_H: i32 = 40;
/// Horizontal gap between layer columns.
pub const COL_GAP: i32 = 72;
/// Vertical gap between node rows within a column.
pub const ROW_GAP: i32 = 28;
/// Canvas padding.
pub const PAD: i32 = 24;
/// Extra height reserved below the nodes for back-edge "rewind" routing
/// (only added when the graph has at least one back-edge).
pub const BACK_LANE: i32 = 44;
