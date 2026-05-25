//! Dimensionality reduction demos. Home for all dim-reduction
//! demo entries -- groups PCA, UMAP, MDS, random projection,
//! and the comparison demos in one file as the milestone
//! progresses. Saga 33 step 030 seeds it with PCA_3D
//! (interactive Plotly 3D viewer).

use crate::demos::Demo;

pub const PCA_3D: Demo = Demo {
    name: "PCA 3D (interactive)",
    intro: "Three-cluster blob dataset projected to its top three principal components, rendered as an interactive [[PCA (Principal Component Analysis)]] viewer via the new `svg(_, \"plotly3d\", labels)` viz type. Drag to rotate, scroll to zoom, double-click to reset. Click any point to see which cluster + sample index it belongs to (the click handler is wired but the REPL panel hookup is a follow-up step -- click events log to the browser console for now).",
    takeaway: "Three well-separated clusters fall into three corners of the 3D embedding. Unlike the static 2D `scatter_labeled` rendering, you can rotate to confirm the separation holds along every axis -- the natural answer to the user question 'are these clusters actually distinct, or is the projection hiding overlap?'. The plot is driven by Plotly via a self-contained HTML/JS payload returned from the new viz type; integrates the existing `pca` builtin without further plumbing.",
    lines: &[
        "# Three 5-D blobs, well-separated centers.",
        "centers = [[0, 0, 0, 0, 0], [5, 5, 5, 5, 5], [-5, 5, -5, 5, -5]]",
        "D = blobs(7, 30, centers)                                                  # 90 points in 5-D, third column-block carries the class label",
        "X = matmul(D, [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,0]]) # 90x5 coordinates only",
        "tl = reshape(matmul(D, [[0],[0],[0],[0],[1]]), [90])                       # integer cluster labels (0/1/2)",
        "# Project to 3 principal components using the existing pca builtin.",
        "P = pca(X, 3)                                                              # 90x3",
        "# Interactive 3D scatter. Drag = rotate, scroll = zoom, click = identify sample.",
        "svg(P, \"plotly3d\", tl)",
    ],
};
