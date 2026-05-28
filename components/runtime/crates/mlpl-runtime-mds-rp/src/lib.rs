//! Saga 33 step 041 (Phase 5 of the dim-reduction milestone):
//! Multidimensional Scaling + Johnson-Lindenstrauss random
//! projection. Sibling crate to mlpl-runtime-dim-reduction
//! (PCA + t-SNE) and mlpl-runtime-umap (UMAP).
//!
//! Each module exposes a `NAMES` constant and a
//! `try_call(name, args)` dispatcher used by mlpl-runtime's
//! registry to delegate. Depends only on mlpl-array and
//! mlpl-runtime-core; nothing depends back upward.

pub mod mds;
pub mod random_projection;
