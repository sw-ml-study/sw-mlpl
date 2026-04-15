Phase 1 step 002: Dataset operations. Add three builtins for preparing data for training.

shuffle(x, seed): returns a row-permutation of a rank >= 1 array using a seeded RNG. The permutation acts only on axis 0; labels are preserved. Deterministic given the seed.

batch(x, size): returns a rank-(r+1) array of contiguous batches along axis 0. If x has shape [N, d1, ..., dk] and size=s, result has shape [ceil(N/s), s, d1, ..., dk]. Short last batch is padded with zeros; add an optional batch_mask(x, size) helper that returns a rank-2 [ceil(N/s), s] 0/1 mask for the valid-row positions. Labels: first axis gets label "batch" if x was labeled along axis 0, second axis inherits that original axis-0 label.

split(x, train_frac, seed): returns a 2-row DenseArray-of-indices or, simpler, returns the training chunk; add a matching val_split(x, train_frac, seed) twin that returns the validation chunk. Both shuffle internally with the same seed so a caller doing train = split(X, 0.8, 7) and val = val_split(X, 0.8, 7) gets disjoint row sets. train_frac is a float in (0, 1).

TDD:
(1) shuffle of a small array is a permutation: contains the same rows, different order, deterministic given seed.
(2) batch of a [10, 2] array with size=3 returns [4, 3, 2] with last batch zero-padded.
(3) batch_mask of the same returns [4, 3] with last row [1, 0, 0] or similar.
(4) split(X, 0.8, seed) + val_split(X, 0.8, seed) cover X disjointly.
(5) labels survive through shuffle and batch.
