//! Saga 33 step 036: dimensionality-reduction tutorial lessons
//! (Phase 4 of the dim-reduction milestone). Six lessons in
//! dependency order, referenced by `lessons::LESSONS`. Lifted
//! into a sibling file so `lessons.rs` and `lessons_advanced.rs`
//! stay under their file-LOC budgets.

use crate::lessons::Lesson;

pub const WHY_REDUCE_DIMENSIONS: Lesson = Lesson {
    title: "Why reduce dimensions?",
    intro: "Two motivations. (1) Visualization: a screen has two coordinates; high-dim data does not. To eyeball a learned embedding -- 'are the cat images close to the dog images, or far?' -- you need a 2-D projection. (2) The manifold hypothesis: real high-D data (images, embeddings, text representations) lies near a much lower-dimensional manifold inside the ambient space. A 768-D ViT embedding might effectively live on a 10-D manifold; dimensionality reduction finds that manifold. The methods divide into LINEAR (PCA: rotate the data along axes of maximum variance) vs MANIFOLD (t-SNE, UMAP: build a local-neighborhood graph, then optimize a low-D layout that preserves it). Linear is fast and exact; manifold methods recover curved structure that linear projections smear. No code in this lesson -- just the framing. The next five lessons walk the methods in order.",
    examples: &[
        "# No code -- this lesson is concept-first.",
        "# The following 5 lessons walk PCA, SNE, t-SNE, UMAP,",
        "# and how to read a critical-dimensions heatmap.",
    ],
    try_it: "Open the 'Dimensionality reduction' learning path from the Paths tab for a curated walk through the demos and glossary entries this lesson points at. The path takes ~15 minutes; each step has a one-line 'why this is here' framing.",
};

pub const PCA_LINEAR_BASELINE: Lesson = Lesson {
    title: "PCA: the linear baseline",
    intro: "PCA finds the directions of maximum variance in your data. Power iteration converges to the dominant eigenvector of the covariance matrix without an eigensolver. MLPL ships three builtins: pca(X, k) returns the projected data [N, k], pca_components(X, k) returns the loadings [k, D] (each row is one principal-component direction in original feature space), and pca_variance_explained(X, k) returns the [k] variance fractions each component captures. Loadings answer a different question than projections: not 'where did the points go?' but 'which input dimensions drive each component?'. The critical-dimensions heatmap viz renders the loadings with per-component variance percentages. PCA is linear -- it cannot recover curved manifolds -- but it is fast, deterministic, and interpretable. Try it first; reach for UMAP only when PCA's smear is unacceptable.",
    examples: &[
        "# Three well-separated 5-D Gaussian clusters.",
        "ca = [0, 0, 0, 0, 0]",
        "cb = [5, 5, 5, 5, 5]",
        "cc = [-5, 5, -5, 5, -5]",
        "pa = randn(1, [30, 5]) * 0.5 + matmul(ones([30, 1]), reshape(ca, [1, 5]))",
        "pb = randn(2, [30, 5]) * 0.5 + matmul(ones([30, 1]), reshape(cb, [1, 5]))",
        "pc = randn(3, [30, 5]) * 0.5 + matmul(ones([30, 1]), reshape(cc, [1, 5]))",
        "X = concat(concat(pa, pb, 0), pc, 0)",
        "# Top-2 PROJECTION: where did the points go?",
        "proj = pca(X, 2)",
        "tl = concat(concat(zeros([30]), ones([30]), 0), ones([30]) + 1, 0)",
        "scatter_labeled(proj, tl)",
        "# Top-3 LOADINGS: which input dims drive each component?",
        "V = pca_components(X, 3)",
        "ve = pca_variance_explained(X, 3)",
        "svg(V, \"critical_dimensions\", ve)",
    ],
    try_it: "Run the 'PCA loadings (critical dimensions)' demo for the full setup. Then change one cluster center so only dim 0 differs (e.g., cb = [5, 0, 0, 0, 0]) and re-render. PC1 should now light up cell 0 strongly (that's the only feature carrying signal) and the other cells should be near zero. The variance-explained percentages also shift: PC1 grabs more, PC2 / PC3 less.",
};

pub const SNE_VERY_SLOW_ANCESTOR: Lesson = Lesson {
    title: "SNE: the very-slow ancestor",
    intro: "Stochastic Neighbor Embedding (Hinton + Roweis, 2002) is t-SNE's predecessor. The setup: high-dim conditional probabilities p_{j|i} = softmax over Gaussian similarities; low-dim conditional probabilities q_{j|i} = softmax over Gaussian similarities again; loss = sum of KL(p_i || q_i). Gradient descent on the low-D coordinates. The math is clean. Two reasons it was abandoned: (1) Asymmetry: p_{j|i} != p_{i|j}, so the loss has no symmetric structure -- the gradient is awkward and the embedding's quality depends on which side of the KL you minimize. (2) Crowding: in low-D, the Gaussian tail decays too fast; well-separated high-D clusters get crushed into one blob because there is not enough 'low-D room' to spread them out. t-SNE fixes both. There is no `sne` builtin -- nobody runs SNE today -- but the failure modes set up why the t-SNE design choices are what they are.",
    examples: &[
        "# No builtin: this lesson describes a method that was",
        "# replaced. Run the t-SNE lesson next to see the fixes.",
        "# The two t-SNE innovations: (1) symmetric p_ij = (p_{j|i} +",
        "# p_{i|j}) / 2N, and (2) Student-t (heavy-tailed) q_ij in",
        "# the low-dim space so well-separated clusters stay separated.",
    ],
    try_it: "Look up Hinton + Roweis 2002 (NeurIPS) if you want the original paper. Then read the t-SNE lesson next to see how van der Maaten + Hinton 2008 fixed asymmetry and the crowding problem in one move.",
};

pub const TSNE_NONLINEAR: Lesson = Lesson {
    title: "t-SNE: a peek at nonlinear methods",
    intro: "t-SNE (van der Maaten + Hinton, 2008) is SNE with two fixes. (1) SYMMETRIZE: p_ij = (p_{j|i} + p_{i|j}) / 2N, so the loss is symmetric and the gradient is clean. (2) HEAVY-TAILED Q: replace the low-dim Gaussian with a Student-t (Cauchy) distribution. The fat tail means well-separated points in high-D can be put far apart in low-D without the gradient pulling them back -- the crowding problem dissolves. The objective is KL(P || Q) with a single P matrix; gradient descent on the low-D coordinates Y. Perplexity sets the per-row scale: it is the 'effective number of neighbors' you want each point to attend to (typical: 5-50). MLPL: tsne(X, perplexity, iters, seed) returns [N, 2]. Caveat: t-SNE's KL is PURELY LOCAL -- it normalizes per row -- so the distance between clusters in the output is meaningless. Cluster SHAPE is meaningful; cluster POSITION is not.",
    examples: &[
        "pa = randn(1, [25, 4]) * 0.5 + matmul(ones([25, 1]), [[0, 0, 0, 0]])",
        "pb = randn(2, [25, 4]) * 0.5 + matmul(ones([25, 1]), [[3, 0, 0, 0]])",
        "pc = randn(3, [25, 4]) * 0.5 + matmul(ones([25, 1]), [[15, 0, 0, 0]])",
        "X = concat(concat(pa, pb, 0), pc, 0)",
        "tl = concat(concat(zeros([25]), ones([25]), 0), ones([25]) + 1, 0)",
        "Y = tsne(X, 10, 200, 1)",
        "scatter_labeled(Y, tl)",
    ],
    try_it: "Rerun with seed = 2 and seed = 3. Cluster shape and relative orientation will rotate / flip -- t-SNE has rotation and reflection ambiguity. Note also that cluster 2 (class label = 2 in the legend) in the input is 5x farther from clusters 0 and 1 than 0 is from 1, but in the t-SNE output you cannot read that ratio. The UMAP lesson next fixes exactly this.",
};

pub const UMAP_MODERN_DEFAULT: Lesson = Lesson {
    title: "UMAP: the modern default",
    intro: "UMAP (McInnes + Healy, 2018) is the modern non-linear dimensionality reduction default. The intuition is Riemannian-geometric: assume the data lies on a smooth manifold; approximate that manifold by a fuzzy simplicial complex -- a local-neighborhood graph whose edge weights are fuzzy-set memberships, calibrated per-point so each point has the same effective Shannon entropy of memberships. Then optimize a low-D layout whose own fuzzy graph is as close as possible to the high-D one, in cross-entropy. The optimization is SGD with negative sampling: per attractive update, sample N_NEG random non-neighbor pairs for the repulsive term. The repulsive term is what gives UMAP its global-distance preservation (t-SNE's KL has no such term -- only local affinities matter). MLPL: umap(X, n_neighbors, min_dist, iters, seed) returns [N, 2]. n_neighbors trades local vs global structure (smaller = more local); min_dist is a soft floor on attractive distances (smaller = tighter clusters).",
    examples: &[
        "# Same three-cluster dataset as the t-SNE lesson.",
        "pa = randn(1, [25, 4]) * 0.5 + matmul(ones([25, 1]), [[0, 0, 0, 0]])",
        "pb = randn(2, [25, 4]) * 0.5 + matmul(ones([25, 1]), [[3, 0, 0, 0]])",
        "pc = randn(3, [25, 4]) * 0.5 + matmul(ones([25, 1]), [[15, 0, 0, 0]])",
        "X = concat(concat(pa, pb, 0), pc, 0)",
        "tl = concat(concat(zeros([25]), ones([25]), 0), ones([25]) + 1, 0)",
        "Y = umap(X, 10, 0.1, 200, 1)",
        "scatter_labeled(Y, tl)",
    ],
    try_it: "Compare with the t-SNE output from the previous lesson on the SAME data. UMAP should put cluster 2 (class label = 2 in the scatter legend) visibly farther from clusters 0 and 1 than 0 is from 1 -- the 5x-input-distance ratio survives (at least partially). Then run the 'UMAP vs t-SNE' demo for the side-by-side comparison and the 'UMAP vs PCA' demo for the manifold-vs-linear case (caveat: MLPL's current UMAP uses a simplified a=1, b=1 Student-t curve and a tight coordinate clamp, so on the moons fixture it separates classes but does not yet preserve the crescent shape -- a follow-up step fixes this). The 'Dim-reduction zoo' demo lays out PCA, t-SNE, UMAP in one row.",
};

pub const READING_CRITICAL_DIMS: Lesson = Lesson {
    title: "Reading a critical-dimensions heatmap",
    intro: "The critical-dimensions viz `svg(V, \"critical_dimensions\", ve)` is k rows (one per component) by D columns (one per input feature). Bright cells = features that dominate that component; dark = features that contribute little. Per-row variance-explained percentages annotate the right margin. The viz is built for PCA loadings but reads cleanly for any [k, D] component / sensitivity matrix. Reading conventions: (1) Components are ORDERED by variance -- PC1 carries the most. If PC1 is dim, dim 1 percentages will be small and the LOADINGS will spread across many features (the data has no dominant direction). (2) Signs are ambiguous: -loading and +loading both contribute, what matters is the magnitude. The viz uses absolute value internally. (3) Permutation sensitivity (a future builtin) reuses the same viz: rows are output dimensions; columns are which input feature was permuted; brightness = how much the output moved. Same v reading rules apply.",
    examples: &[
        "# Same three-cluster 5-D data; this time look at PC loadings.",
        "ca = [0, 0, 0, 0, 0]",
        "cb = [5, 5, 5, 5, 5]",
        "cc = [-5, 5, -5, 5, -5]",
        "pa = randn(1, [30, 5]) * 0.5 + matmul(ones([30, 1]), reshape(ca, [1, 5]))",
        "pb = randn(2, [30, 5]) * 0.5 + matmul(ones([30, 1]), reshape(cb, [1, 5]))",
        "pc = randn(3, [30, 5]) * 0.5 + matmul(ones([30, 1]), reshape(cc, [1, 5]))",
        "X = concat(concat(pa, pb, 0), pc, 0)",
        "V = pca_components(X, 3)",
        "ve = pca_variance_explained(X, 3)",
        "svg(V, \"critical_dimensions\", ve)",
    ],
    try_it: "Bind one cluster's signal to a single dimension: change `cb` to `[5, 0, 0, 0, 0]` and `cc` to `[0, 0, 0, 0, 5]`. Re-run. PC1 should now light up dim 0 strongly (it's what separates cluster 1 from 0); PC2 should light up dim 4 (separates cluster 2). Variance-explained percentages should also concentrate: PC1 and PC2 carry the cluster signal; PC3 is noise. The heatmap teaches you which features the projection 'used.'",
};
