Add a PCA demo that visualizes the principal axis on a 2D synthetic dataset.

1. demos/pca.mlpl: build a small correlated 2D dataset (e.g. y = 2x + noise via randn). Center the data by subtracting the column means. Compute the 2x2 covariance matrix as (X^T X) / n. Find the dominant eigenvector via 10 iterations of power iteration starting from a unit vector. Project the data onto the first principal component to get a 1D coordinate, then render: (a) the original points colored by their PC1 coordinate (use scatter_labeled with discretized labels) and (b) the principal axis as a line through the centroid via svg with a 2x2 polyline matrix.
2. Add PCA to the web demo dropdown (alphabetized).
3. Tutorial lesson "Dimensionality Reduction: PCA" introducing covariance, eigenvectors, and projection.
4. Rebuild pages.
5. Allowed: demos/, apps/mlpl-web/src/, docs/