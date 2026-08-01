//! The n-gram rolling-hash reference contract. Every backend (CPU
//! pipeline, MLX, later CUDA) must reproduce these indices
//! bit-for-bit -- that is the E5 acceptance gate. The reference
//! computes in u64, but every intermediate is constrained below
//! 2^53 so an f64 tensor pipeline lands on identical values
//! (decision D4 in docs/engram-sagas-plan.md).

use mlpl_engram_core::{EngramError, HashSpec, MAX_TOKEN_ID, head_offset, ngram_hashes};

fn spec() -> HashSpec {
    HashSpec {
        ngram_orders: vec![2, 3],
        heads_per_ngram: 4,
        slots_per_head: 1024,
        seed: 7,
    }
}

#[test]
fn shape_is_positions_by_order_by_head() {
    let h = ngram_hashes(&[10, 20, 30, 40], &spec()).unwrap();
    assert_eq!(h.len(), 4, "one entry per position");
    assert_eq!(h[0].len(), 2, "one entry per n-gram order");
    assert_eq!(h[0][0].len(), 4, "one entry per head");
}

#[test]
fn all_indices_stay_inside_the_table() {
    let s = spec();
    let h = ngram_hashes(&[0, 1, 2, 3, 4, 5, 1_000_000, 2_097_151], &s).unwrap();
    for t in &h {
        for o in t {
            for &slot in o {
                assert!(slot < s.slots_per_head as u64);
            }
        }
    }
}

#[test]
fn deterministic_and_seed_sensitive() {
    let a = ngram_hashes(&[10, 20, 30, 40], &spec()).unwrap();
    let b = ngram_hashes(&[10, 20, 30, 40], &spec()).unwrap();
    assert_eq!(a, b, "same seed, same ids => identical");
    let mut other = spec();
    other.seed = 8;
    let c = ngram_hashes(&[10, 20, 30, 40], &other).unwrap();
    assert_ne!(a, c, "different seed must move the indices");
}

#[test]
fn heads_disagree() {
    let h = ngram_hashes(&[10, 20, 30, 40], &spec()).unwrap();
    let last = &h[3][0];
    assert!(
        last.windows(2).any(|w| w[0] != w[1]),
        "independent heads should not all collide: {last:?}"
    );
}

#[test]
fn bigram_window_is_two_trigram_window_is_three() {
    let s = spec();
    let base = ngram_hashes(&[10, 20, 30, 40], &s).unwrap();
    // Changing ids[0] must NOT change the BIGRAM hash at t=3
    // (window is ids[2..=3]) but MUST change the TRIGRAM hash at
    // t=2 (window is ids[0..=2]).
    let moved = ngram_hashes(&[11, 20, 30, 40], &s).unwrap();
    assert_eq!(base[3][0], moved[3][0], "bigram at t=3 ignores ids[0]");
    assert_ne!(base[2][1], moved[2][1], "trigram at t=2 sees ids[0]");
}

#[test]
fn leading_positions_use_pad_zero() {
    // At t=0 a bigram has no previous token; the reference defines
    // the missing history as PAD id 0, so [X] hashes like [0, X].
    let s = spec();
    let one = ngram_hashes(&[10], &s).unwrap();
    let two = ngram_hashes(&[0, 10], &s).unwrap();
    assert_eq!(one[0][0], two[1][0], "implicit and explicit pad agree");
}

#[test]
fn token_ids_beyond_the_f64_exact_bound_are_rejected() {
    let err = ngram_hashes(&[MAX_TOKEN_ID + 1], &spec()).unwrap_err();
    assert!(matches!(err, EngramError::TokenIdTooLarge { .. }));
}

#[test]
fn head_offsets_partition_the_flat_table() {
    let s = spec();
    assert_eq!(head_offset(&s, 0, 0), 0);
    assert_eq!(head_offset(&s, 0, 1), 1024);
    assert_eq!(head_offset(&s, 1, 0), 4 * 1024);
    assert_eq!(head_offset(&s, 1, 3), 7 * 1024);
}

#[test]
fn f64_pipeline_mirrors_the_u64_reference_exactly() {
    // Recompute one hash the way the tensor pipeline will: pure
    // f64 arithmetic. Exactness holds because every product and
    // sum stays below 2^53.
    let s = spec();
    let ids: Vec<u64> = vec![10, 20, 30, 2_097_151];
    let reference = ngram_hashes(&ids, &s).unwrap();
    let mults = mlpl_engram_core::hash_multipliers(&s);
    // All values fit u32, so f64::from is lossless -- mirroring the
    // exactness argument itself.
    let lossless = |v: u64| f64::from(u32::try_from(v).expect("bounded by contract"));
    for (t, _) in ids.iter().enumerate() {
        for (oi, &order) in s.ngram_orders.iter().enumerate() {
            for head in 0..s.heads_per_ngram {
                let mut acc = 0f64;
                for k in 0..order {
                    let id = lossless(if t >= k { ids[t - k] } else { 0 });
                    let prod = id * lossless(mults[oi][head][k]);
                    acc += prod % 2_147_483_647f64;
                }
                let slot = acc % lossless(s.slots_per_head as u64);
                // Bit-level equality: the strongest form of the
                // exactness claim (also satisfies clippy float_cmp).
                assert_eq!(
                    slot.to_bits(),
                    lossless(reference[t][oi][head]).to_bits(),
                    "f64 pipeline must land on the reference at t={t} o={oi} h={head}"
                );
            }
        }
    }
}

#[test]
fn golden_fixture_v1() {
    // FROZEN: the cross-backend parity fixture (ids [10,20,30,40],
    // seed 7, orders [2,3], 4 heads, 1024 slots). Any change to
    // these numbers is a BREAKING change to every trained
    // checkpoint and backend -- bump a format version instead.
    let h = ngram_hashes(&[10, 20, 30, 40], &spec()).unwrap();
    let flat: Vec<u64> = h.into_iter().flatten().flatten().collect();
    insta_like_assert(&flat);
}

/// Golden comparison helper: panics with the actual values so a
/// first run can freeze them, then guards them forever.
fn insta_like_assert(actual: &[u64]) {
    const FROZEN: &[u64] = &[
        587, 896, 82, 144, 311, 59, 156, 649, 183, 409, 308, 800, 622, 425, 600, 521, 804, 946,
        534, 432, 503, 360, 946, 842, 401, 460, 760, 65, 383, 297, 268, 141,
    ];
    assert_eq!(actual, FROZEN, "golden hash fixture v1 drifted");
}

#[test]
fn addressing_stats_count_lookups_rows_and_collisions() {
    use mlpl_engram_core::addressing_stats;
    // 4 positions x 2 orders x 4 heads = 32 lookups; unique rows
    // and collisions derived from the same frozen hashes.
    let ids = [10u64, 20, 30, 40];
    let s = addressing_stats(&ids, &spec()).unwrap();
    assert_eq!(s.lookups, 32);
    let h = ngram_hashes(&ids, &spec()).unwrap();
    let mut rows = std::collections::BTreeSet::new();
    for (t, per_order) in h.iter().enumerate() {
        let _ = t;
        for (oi, heads) in per_order.iter().enumerate() {
            for (head, &local) in heads.iter().enumerate() {
                rows.insert(head_offset(&spec(), oi, head) + local);
            }
        }
    }
    assert_eq!(s.unique_rows, rows.len() as u64);
    // All 4 contexts are distinct per (order, head) and 1024 slots
    // make actual collisions vanishingly unlikely at this size.
    assert_eq!(s.collisions, 0);
}

#[test]
fn addressing_stats_repetition_is_not_collision_but_crowding_is() {
    use mlpl_engram_core::addressing_stats;
    let tiny = HashSpec {
        ngram_orders: vec![2],
        heads_per_ngram: 1,
        slots_per_head: 2,
        seed: 11,
    };
    // Five distinct bigram contexts into 2 slots: collisions =
    // contexts - distinct rows.
    let s = addressing_stats(&[1, 2, 3, 4, 5], &tiny).unwrap();
    assert_eq!(s.lookups, 5);
    assert!(s.unique_rows <= 2);
    assert_eq!(s.collisions, 5 - s.unique_rows);
    // Repeated identical context (5,5) is repetition, not collision.
    let wide = HashSpec {
        ngram_orders: vec![2],
        heads_per_ngram: 1,
        slots_per_head: 64,
        seed: 7,
    };
    let r = addressing_stats(&[5, 5, 5], &wide).unwrap();
    assert_eq!(r.unique_rows, 2);
    assert_eq!(r.collisions, 0);
}
