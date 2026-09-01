//! Typed model for the dataflow renderer: a validated graph and its
//! laid-out positions. Phase 1 carries node labels + directed edges
//! (with optional per-edge labels); groups / widths / highlight are
//! later phases (docs/dataflow-renderer-design.md).

/// A validated dataflow graph. Node ids are indices into `labels`;
/// every edge endpoint is in range and `edge_labels` is either empty
/// or one-per-edge (guaranteed by `build`).
pub struct Graph {
    /// One box label per node; index = node id.
    pub labels: Vec<String>,
    /// Directed edges as `(from_id, to_id)` pairs.
    pub edges: Vec<(usize, usize)>,
    /// Per-edge label, or empty for none.
    pub edge_labels: Vec<String>,
}

/// A graph with a pixel position assigned to every node, plus the
/// canvas size the render step draws into.
pub struct Positioned {
    /// The source graph.
    pub graph: Graph,
    /// Top-left `(x, y)` of each node box, indexed by node id.
    pub pos: Vec<(i32, i32)>,
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
