//! Dimensionality reduction demos. Home for all dim-reduction
//! demo entries -- groups PCA, UMAP, MDS, random projection,
//! and the comparison demos in one file as the milestone
//! progresses. Saga 33 step 030 seeded it with PCA_3D; step
//! 032 adds PCA_LOADINGS for the critical-dimensions viz.

use crate::demos::Demo;

pub const PCA_3D: Demo = Demo {
    name: "PCA 3D (interactive)",
    intro: "Three 5-D Gaussian clusters projected to their top three principal components and rendered as an interactive [[PCA (Principal Component Analysis)]] viewer via the new `svg(_, \"plotly3d\", labels)` viz type. Drag to rotate, scroll to zoom, double-click to reset. Click any point and the browser console logs which sample + cluster it belongs to (the click handler is wired but the REPL panel hookup is a follow-up step).",
    takeaway: "Three well-separated clusters fall into three corners of the 3D PCA embedding. Unlike the static 2D `scatter_labeled` rendering, you can rotate to confirm the separation holds along every axis -- the answer to the user question 'are these clusters actually distinct, or is the projection hiding overlap?'. The plot is driven by Plotly via a self-contained HTML/JS payload returned from the new viz type; integrates the existing `pca` builtin without further plumbing.",
    lines: &[
        "# Build 90 noisy points in 5-D from three cluster centers.",
        "# `blobs` only supports 2-D, so we hand-roll the construction",
        "# via randn + broadcasted centers + concat.",
        "center_a = [0, 0, 0, 0, 0]",
        "center_b = [5, 5, 5, 5, 5]",
        "center_c = [-5, 5, -5, 5, -5]",
        "noise = 0.5",
        "pts_a = randn(1, [30, 5]) * noise + matmul(ones([30, 1]), reshape(center_a, [1, 5]))",
        "pts_b = randn(2, [30, 5]) * noise + matmul(ones([30, 1]), reshape(center_b, [1, 5]))",
        "pts_c = randn(3, [30, 5]) * noise + matmul(ones([30, 1]), reshape(center_c, [1, 5]))",
        "X = concat(concat(pts_a, pts_b, 0), pts_c, 0)                                # 90 x 5",
        "tl = concat(concat(zeros([30]), ones([30]), 0), ones([30]) + 1, 0)            # cluster labels: 30 zeros, 30 ones, 30 twos",
        "# Project the 5-D data onto its top 3 principal components.",
        "P = pca(X, 3)                                                                  # 90 x 3",
        "# Interactive 3D scatter. Drag = rotate, scroll = zoom, click = identify sample.",
        "svg(P, \"plotly3d\", tl)",
    ],
};

pub const PCA_LOADINGS: Demo = Demo {
    name: "PCA loadings (critical dimensions)",
    intro: "Same three-cluster 5-D dataset as the PCA 3D demo, but instead of plotting the projected points we plot the LOADINGS -- the directions in original feature space that each principal component aligns with. This is the [[PCA (Principal Component Analysis)]] answer to 'which input dimensions matter?'. The new `pca_components(X, k)` builtin returns the `[k, D]` loadings matrix; the new `pca_variance_explained(X, k)` builtin returns the per-component variance fractions. The `svg(_, \"critical_dimensions\", _)` viz renders both as a heatmap with PC labels.",
    takeaway: "Each row is one principal component (PC1, PC2, PC3); each column is one input dimension (0..4). Bright cells = features that dominate that component; dark cells = features that contribute little. With three well-separated clusters in 5-D, the top two components usually capture >90% of the variance (see the per-row percentages on the right) and pull from a handful of features each. The same `critical_dimensions` viz will be reused in the dim-reduction milestone for UMAP / t-SNE permutation-sensitivity heatmaps once those land.",
    lines: &[
        "# Same 5-D cluster build as the PCA 3D demo.",
        "center_a = [0, 0, 0, 0, 0]",
        "center_b = [5, 5, 5, 5, 5]",
        "center_c = [-5, 5, -5, 5, -5]",
        "noise = 0.5",
        "pts_a = randn(1, [30, 5]) * noise + matmul(ones([30, 1]), reshape(center_a, [1, 5]))",
        "pts_b = randn(2, [30, 5]) * noise + matmul(ones([30, 1]), reshape(center_b, [1, 5]))",
        "pts_c = randn(3, [30, 5]) * noise + matmul(ones([30, 1]), reshape(center_c, [1, 5]))",
        "X = concat(concat(pts_a, pts_b, 0), pts_c, 0)",
        "# Top-3 principal-component LOADINGS (directions in original 5-D feature space).",
        "V = pca_components(X, 3)                                          # [3, 5]",
        "# Per-component variance-explained ratios.",
        "ve = pca_variance_explained(X, 3)                                 # [3]",
        "# Critical-dimensions heatmap with per-row variance percentages.",
        "svg(V, \"critical_dimensions\", ve)",
    ],
};
