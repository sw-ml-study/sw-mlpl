Implement k-means clustering as an MLPL demo and wire it into the web REPL.

1. demos/kmeans.mlpl: generate a blobs dataset (3 classes, 30 points each), initialize 3 cluster centers (e.g. first three points), then run a fixed number of Lloyd iterations. Each iteration: assign every point to its nearest center (use a vectorized distance computation), then move each center to the mean of its assigned points. After convergence, render the final clustering with scatter_labeled and overlay the centers using svg(centers, "scatter").
2. The demo should run end-to-end via cargo run -p mlpl-repl -- -f demos/kmeans.mlpl with no errors.
3. Add Kmeans as a Web demo entry (alphabetized).
4. Add a tutorial lesson "Unsupervised: K-Means" walking through the assignment/update split at a beginner level.
5. Rebuild pages.
6. Allowed: demos/, apps/mlpl-web/src/, docs/