//! The n-gram rolling-hash REFERENCE implementation. This is the
//! contract every backend must match bit-for-bit (the saga E5
//! CPU==MLX acceptance gate hashes against these functions).
//!
//! Exact-arithmetic design (decision D4, docs/engram-sagas-plan.md):
//! the reference computes in u64, but every intermediate is
//! constrained below 2^53 so a pure-f64 tensor pipeline lands on
//! IDENTICAL values:
//!
//! - token ids are capped at [`crate::MAX_TOKEN_ID`] (2^21 - 1),
//! - multipliers live in `[1, P)` with `P = 2^31 - 1` (prime), so
//!   every product `id * m < 2^52`,
//! - each product is reduced `mod P` (< 2^31) before summing, so a
//!   sum over an n-gram window stays far below 2^53,
//! - the accumulated sum reduces `mod slots_per_head`.
//!
//! Missing history at the sequence start is PAD id `0`.

use std::collections::BTreeSet;

use crate::spec::EngramError;

/// The prime modulus every product is reduced by. FROZEN: part of
/// the cross-backend hash contract (2^31 - 1).
pub const HASH_PRIME: u64 = 2_147_483_647;

/// Hash-relevant subset of an Engram configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashSpec {
    pub ngram_orders: Vec<usize>,
    pub heads_per_ngram: usize,
    pub slots_per_head: usize,
    pub seed: u64,
}

/// Deterministic per-(order, head, window-position) multipliers in
/// `[1, HASH_PRIME)`, derived from the seed via splitmix64. Shaped
/// `[order_idx][head][k]` where `k` indexes the window position
/// (0 = current token, 1 = previous, ...).
#[must_use]
pub fn hash_multipliers(spec: &HashSpec) -> Vec<Vec<Vec<u64>>> {
    let splitmix = |state: u64| -> u64 {
        let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let base = spec.seed.wrapping_mul(0x0100_0000_01B3);
    spec.ngram_orders
        .iter()
        .enumerate()
        .map(|(oi, &order)| {
            (0..spec.heads_per_ngram)
                .map(|head| {
                    (0..order)
                        .map(|k| {
                            let tag = ((oi as u64) << 40) | ((head as u64) << 20) | k as u64;
                            splitmix(base.wrapping_add(tag)) % (HASH_PRIME - 1) + 1
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Rolling n-gram hashes for a token sequence: LOCAL slot indices
/// shaped `[position][order_idx][head]`, each `< slots_per_head`.
///
/// # Errors
/// [`EngramError::TokenIdTooLarge`] when any id exceeds the
/// exact-arithmetic bound.
pub fn ngram_hashes(ids: &[u64], spec: &HashSpec) -> Result<Vec<Vec<Vec<u64>>>, EngramError> {
    if let Some(&id) = ids.iter().find(|&&id| id > crate::MAX_TOKEN_ID) {
        return Err(EngramError::TokenIdTooLarge { id });
    }
    let (mults, slots) = (hash_multipliers(spec), spec.slots_per_head as u64);
    Ok((0..ids.len())
        .map(|t| {
            spec.ngram_orders
                .iter()
                .enumerate()
                .map(|(oi, &order)| {
                    (0..spec.heads_per_ngram)
                        .map(|head| {
                            (0..order)
                                .map(|k| if t >= k { ids[t - k] } else { 0 })
                                .zip(&mults[oi][head])
                                .fold(0, |s, (id, m)| s + (id * m) % HASH_PRIME)
                                % slots
                        })
                        .collect()
                })
                .collect()
        })
        .collect())
}

/// Row offset of `(order_idx, head)`'s region in the ONE flattened
/// memory table: `(order_idx * heads + head) * slots_per_head`.
/// `global_index = head_offset(..) + local_hash`.
#[must_use]
pub fn head_offset(spec: &HashSpec, order_idx: usize, head: usize) -> u64 {
    ((order_idx * spec.heads_per_ngram + head) * spec.slots_per_head) as u64
}

/// Addressing statistics for a token sequence under the frozen
/// hash contract (saga E3: the `engram_stats` builtin's counters).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddressingStats {
    /// Total `(position, order, head)` lookups.
    pub lookups: u64,
    /// Distinct global memory rows addressed.
    pub unique_rows: u64,
    /// Distinct n-gram contexts forced to SHARE a row with a
    /// different context, summed per `(order, head)` region. A
    /// repeated identical context is repetition, not collision.
    pub collisions: u64,
}

/// Compute [`AddressingStats`] for `ids`. Contexts use the same
/// PAD-0 windowing as [`ngram_hashes`], so `collisions` is exact:
/// per region, `distinct contexts - distinct rows`.
///
/// # Errors
/// Same surface as [`ngram_hashes`] (oversized token ids).
pub fn addressing_stats(ids: &[u64], spec: &HashSpec) -> Result<AddressingStats, EngramError> {
    let hashes = ngram_hashes(ids, spec)?;
    let pad = |t: usize, k: usize| if t >= k { ids[t - k] } else { 0 };
    let (mut all_rows, mut collisions) = (BTreeSet::new(), 0u64);
    for (oi, &order) in spec.ngram_orders.iter().enumerate() {
        let contexts: BTreeSet<Vec<u64>> = (0..ids.len())
            .map(|t| (0..order).map(|k| pad(t, k)).collect())
            .collect();
        let mut rows_per_head = vec![BTreeSet::new(); spec.heads_per_ngram];
        for t_hashes in &hashes {
            for (head, (&local, rows)) in t_hashes[oi].iter().zip(&mut rows_per_head).enumerate() {
                rows.insert(head_offset(spec, oi, head) + local);
            }
        }
        for rows in &rows_per_head {
            all_rows.extend(rows.iter().copied());
            collisions += (contexts.len() - rows.len()) as u64;
        }
    }
    Ok(AddressingStats {
        lookups: (ids.len() * spec.ngram_orders.len() * spec.heads_per_ngram) as u64,
        unique_rows: all_rows.len() as u64,
        collisions,
    })
}
