//! Unsupervised-exploration demos: dimensionality reduction
//! (PCA, UMAP, MDS, random projection + comparisons) plus
//! K-Means clustering (category "Clustering"). Grouped here as
//! the "learn structure from unlabeled data" family. This file
//! was seeded for dim-reduction as the milestone
//! progresses. Saga 33 step 030 seeded it with PCA_3D; step
//! 032 added PCA_LOADINGS for the critical-dimensions viz;
//! step 035 adds the UMAP_VS_PCA / UMAP_VS_TSNE / DIM_REDUCTION_ZOO
//! comparison demos and registers a dedicated
//! "Dimensionality reduction" path/category.

use mlpl_web_demos_types::Demo;

pub const PCA_3D: Demo = Demo {
    category: "Dim Reduction",
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
    category: "Dim Reduction",
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

pub const UMAP_VS_PCA: Demo = Demo {
    category: "Dim Reduction",
    name: "UMAP vs PCA",
    intro: "Two-moons in 2-D, embedded in 5-D with three low-variance noise dimensions, then projected back to 2-D by both [[PCA (Principal Component Analysis)]] (linear) and [[UMAP]] (non-linear / manifold-aware). PCA picks the top two axes of variance; UMAP builds a k-NN graph over the full 5-D distances and runs SGD on the fuzzy simplicial set with `a, b` curve params fitted from `min_dist`.",
    takeaway: "PCA reconstructs the moon arcs because the noise dimensions carry less variance than the moons themselves -- you can see crescents along PC1/PC2. UMAP recovers the two moons from local neighborhood structure: class 0 lands in one region of the embedding, class 1 in the other, with minimal cross-mixing -- a different recipe for the same end result. The legend in each scatter maps integer class id to color: 0 = blue, 1 = pink.",
    lines: &[
        "# Two-moons (100 points, light noise) is the test bed.",
        "M = moons(7, 100, 0.05)                                            # 100 x 3: x, y, label",
        "moons2d = matmul(M, [[1, 0], [0, 1], [0, 0]])                       # drop label column",
        "labels = reshape(matmul(M, [[0], [0], [1]]), [100])                # integer labels",
        "# Embed in 5-D with three low-variance noise dimensions.",
        "embed = randn(42, [100, 3]) * 0.3",
        "X = concat(moons2d, embed, 1)                                      # 100 x 5",
        "# PCA: top 2 linear principal components.",
        "pca_proj = pca(X, 2)",
        "# UMAP: k-NN graph + fuzzy simplicial set + layout SGD.",
        "umap_proj = umap(X, 15, 0.1, 200, 7)                                # n_neighbors, min_dist, iters, seed",
        "# Side by side, colored by class.",
        "scatter_labeled(pca_proj, labels)",
        "scatter_labeled(umap_proj, labels)",
    ],
};

pub const UMAP_VS_TSNE: Demo = Demo {
    category: "Dim Reduction",
    name: "UMAP vs t-SNE",
    intro: "Three clusters in 4-D where cluster 2 is five times farther from clusters 0 and 1 than cluster 0 is from cluster 1. The legend in each scatter maps integer class id to color: 0 = blue (origin), 1 = pink (near), 2 = green (far). Both [[t-SNE]] and [[UMAP]] use a fuzzy-graph view of local neighborhoods, but they handle GLOBAL inter-cluster distance differently. The comparison shows what t-SNE drops and UMAP keeps.",
    takeaway: "t-SNE tends to inflate every cluster to a similar size, washing out 'cluster 2 is much farther than 0 is from 1.' UMAP's repulsive force keeps the relative distances readable -- cluster 2 (green) ends up clearly farther from {0, 1} than 0 is from 1. The structural reason: t-SNE's KL objective is purely local (it normalizes per row), while UMAP's cross-entropy + negative-sampling objective lets the repulsive term carry global signal.",
    lines: &[
        "# Three 4-D Gaussian clusters: A at origin, B near A, C 5x farther.",
        "pts_a = randn(1, [30, 4]) * 0.5 + matmul(ones([30, 1]), [[0, 0, 0, 0]])",
        "pts_b = randn(2, [30, 4]) * 0.5 + matmul(ones([30, 1]), [[3, 0, 0, 0]])",
        "pts_c = randn(3, [30, 4]) * 0.5 + matmul(ones([30, 1]), [[15, 0, 0, 0]])",
        "X = concat(concat(pts_a, pts_b, 0), pts_c, 0)                                       # 90 x 4",
        "labels = concat(concat(zeros([30]), ones([30]), 0), ones([30]) + 1, 0)              # 30 zeros, 30 ones, 30 twos",
        "# t-SNE: perplexity-calibrated local KL, no global anchor.",
        "tsne_proj = tsne(X, 10, 200, 1)",
        "# UMAP: fuzzy simplicial set + cross-entropy with negative sampling.",
        "umap_proj = umap(X, 10, 0.1, 200, 1)",
        "scatter_labeled(tsne_proj, labels)",
        "scatter_labeled(umap_proj, labels)",
    ],
};

pub const DIM_REDUCTION_ZOO: Demo = Demo {
    category: "Dim Reduction",
    name: "Dim-reduction zoo",
    intro: "[[PCA (Principal Component Analysis)]], [[t-SNE]], [[UMAP]], [[Multidimensional Scaling]], and [[Johnson-Lindenstrauss Lemma]] random projection -- five methods on the same 5-D three-cluster dataset, rendered as a row of five scatter plots for direct comparison.",
    takeaway: "PCA is the cheap linear baseline that preserves the global axes of variance -- clusters land along PC1/PC2 directions. t-SNE inflates each cluster to roughly equal radius, sharpening local boundaries at the cost of global distance. UMAP keeps both: tight clusters AND the relative positions of those clusters in the original 5-D feature space. MDS preserves pairwise distances directly (the answer to 'which points are far from which?'). Random projection is the JL sanity baseline -- a Gaussian random matrix preserves all pairwise distances within a (1 +- eps) factor by sheer probability, no optimization needed; if a learned method does not beat random projection, the learned features are not adding signal.",
    lines: &[
        "# Three 5-D Gaussian clusters at distinct locations.",
        "pts_a = randn(1, [25, 5]) * 0.4 + matmul(ones([25, 1]), [[0, 0, 0, 0, 0]])",
        "pts_b = randn(2, [25, 5]) * 0.4 + matmul(ones([25, 1]), [[3, 3, 0, 0, 0]])",
        "pts_c = randn(3, [25, 5]) * 0.4 + matmul(ones([25, 1]), [[0, 0, 3, 3, 3]])",
        "X = concat(concat(pts_a, pts_b, 0), pts_c, 0)                                       # 75 x 5",
        "labels = concat(concat(zeros([25]), ones([25]), 0), ones([25]) + 1, 0)",
        "# Linear baseline.",
        "pca_proj = pca(X, 2)",
        "# Local-only neighborhood projection.",
        "tsne_proj = tsne(X, 8, 150, 1)",
        "# Local + global projection.",
        "umap_proj = umap(X, 8, 0.1, 150, 1)",
        "# Preserves pairwise distances directly.",
        "mds_proj = mds(X, 2, 80, 1)",
        "# Johnson-Lindenstrauss random-projection sanity baseline.",
        "rp_proj = random_projection(X, 2, 7)",
        "scatter_labeled(pca_proj, labels)",
        "scatter_labeled(tsne_proj, labels)",
        "scatter_labeled(umap_proj, labels)",
        "scatter_labeled(mds_proj, labels)",
        "scatter_labeled(rp_proj, labels)",
    ],
};

pub const KMEANS: Demo = Demo {
    category: "Clustering",
    name: "K-Means",
    intro: "K-means clustering without loops over points: all distances computed in one matmul, cluster assignments via argmax, and centroid updates via a one-hot-matrix-times-data trick. Three blobs in 2D, three centroids, ten iterations.",
    takeaway: "The final scatter shows points colored by their assigned cluster and the three centroids as a separate plot. Unsupervised -- no labels were passed in; the algorithm discovered the three groups from geometry alone.",
    lines: &[
        "D = blobs(7, 30, [[0, 0], [4, 4], [-4, 4]])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # 90 points across three blobs (with labels)",
        "X = matmul(D, [[1,0],[0,1],[0,0]])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # drop the label column for unsupervised work",
        "C = [[1, 1], [3, 3], [-3, 3]]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       # 3 initial centroids",
        "repeat 10 { sqX = reshape(reduce_add(X*X, 1), [90, 1]); sqC = reshape(reduce_add(C*C, 1), [1, 3]); XC = matmul(X, transpose(C)); dists = matmul(sqX, ones([1, 3])) + matmul(ones([90, 1]), sqC) - 2*XC; clus = argmax(-1 * dists, 1); jj = reshape(range(3), [3, 1]); ll = reshape(clus, [1, 90]); diff = matmul(jj, ones([1, 90])) - matmul(ones([3, 1]), ll); A = eq(diff, 0); counts = reshape(reduce_add(A, 1), [3, 1]); sums = matmul(A, X); C = sums / matmul(counts, ones([1, 2])) }                                                                                                                                                                                              # 10 iterations of assign + update -- no per-point loop",
        "sqX = reshape(reduce_add(X*X, 1), [90, 1])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # per-point squared norms (final assignment)",
        "sqC = reshape(reduce_add(C*C, 1), [1, 3])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # per-centroid squared norms",
        "XC = matmul(X, transpose(C))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       # cross-products",
        "dists = matmul(sqX, ones([1, 3])) + matmul(ones([90, 1]), sqC) - 2*XC                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # squared-distance matrix [90, 3]",
        "clus = argmax(-1 * dists, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       # nearest centroid per point (argmax of negative distance)",
        "scatter_labeled(X, clus)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            # points colored by cluster",
        "svg(C, \"scatter\")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  # final centroid positions",
    ],
};

pub const PCA: Demo = Demo {
    category: "Dim Reduction",
    name: "PCA",
    intro: "Principal Component Analysis without calling into a library: make anisotropic 2D data, center it, form the covariance matrix, run [[power iteration]] to find the top eigenvector, and project every point onto it. The top axis is the direction of maximum variance.",
    takeaway: "The scatter is colored by which side of zero each point's projection lands on; the line shows the found principal axis. Power iteration converges in ~10 steps to a direction that's clearly the long axis of the data cloud.",
    lines: &[
        "Xraw = randn(1, [60, 2])                                          # 60 points of isotropic 2D noise",
        "X = matmul(Xraw, [[1, 2], [0, 0.3]])                              # stretch into an anisotropic cloud",
        "cm = reduce_add(X, 0) / 60                                        # column means -- the centroid",
        "Xc = X - matmul(ones([60, 1]), reshape(cm, [1, 2]))               # center the data",
        "Cov = matmul(transpose(Xc), Xc) / 60                              # 2x2 covariance matrix",
        "v = [1, 0]                                                         # initial guess for the top eigenvector",
        "repeat 10 { v = matmul(Cov, v); v = v / sqrt(dot(v, v)) }         # power iteration: 10 multiplies + normalize",
        "coords = reshape(matmul(Xc, reshape(v, [2, 1])), [60])            # project each point onto the principal axis",
        "labels = gt(coords, 0)                                             # which side of the axis each point lands on",
        "ends = matmul(reshape([-3, 3], [2, 1]), reshape(v, [1, 2]))       # endpoints of the axis line in centered space",
        "line = ends + matmul(ones([2, 1]), reshape(cm, [1, 2]))           # shift the line back into original space",
        "scatter_labeled(X, labels)                                         # data colored by side of the principal axis",
        "svg(line, \"line\")                                                 # the principal axis itself",
    ],
};
