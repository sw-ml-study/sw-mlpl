//! `EngramSpec` validation + derived accounting. The accounting
//! numbers pin the `:describe` example from
//! docs/engram-support-in-sw-mlpl.txt so the REPL introspection
//! (saga E2) can never drift from the core arithmetic.

use mlpl_engram_core::{DType, EngramError, EngramSpec};

fn demo_spec() -> EngramSpec {
    // The doc's `:describe e` example: ngrams [2,3], 4 heads,
    // 65536 slots, head_dim 16.
    EngramSpec {
        hidden_size: 256,
        ngram_orders: vec![2, 3],
        heads_per_ngram: 4,
        slots_per_head: 65_536,
        head_dim: 16,
        seed: 42,
    }
}

#[test]
fn valid_spec_passes() {
    assert!(demo_spec().validate().is_ok());
}

#[test]
fn rejects_empty_ngrams_and_low_orders() {
    let mut s = demo_spec();
    s.ngram_orders = vec![];
    assert!(matches!(s.validate(), Err(EngramError::EmptyNgramOrders)));
    let mut s = demo_spec();
    s.ngram_orders = vec![1];
    assert!(matches!(
        s.validate(),
        Err(EngramError::NgramOrderTooSmall { order: 1 })
    ));
}

#[test]
fn rejects_zero_dimensions() {
    for field in ["heads", "slots", "head_dim", "hidden"] {
        let mut s = demo_spec();
        match field {
            "heads" => s.heads_per_ngram = 0,
            "slots" => s.slots_per_head = 0,
            "head_dim" => s.head_dim = 0,
            _ => s.hidden_size = 0,
        }
        assert!(s.validate().is_err(), "{field}=0 must be rejected");
    }
}

#[test]
fn rejects_parameter_overflow() {
    let mut s = demo_spec();
    s.slots_per_head = usize::MAX / 2;
    s.head_dim = usize::MAX / 2;
    assert!(matches!(s.validate(), Err(EngramError::Overflow)));
}

#[test]
fn accounting_matches_the_doc_example() {
    let s = demo_spec();
    // 2 orders x 4 heads x 65536 slots = 524,288 table rows.
    assert_eq!(s.table_rows().unwrap(), 524_288);
    // 2 x 4 x 16 = 128 retrieved width.
    assert_eq!(s.retrieved_width().unwrap(), 128);
    // rows x head_dim = 8,388,608 memory parameters.
    assert_eq!(s.parameter_count().unwrap(), 8_388_608);
    // fp16: 16 MiB exactly.
    assert_eq!(s.bytes_for(DType::F16).unwrap(), 16 * 1024 * 1024);
}

#[test]
fn accounting_matches_the_conservative_apple_config() {
    // Doc example A per injected layer: 2 orders x 4 heads x
    // 262144 slots x 32 dims = 67,108,864 params; fp16 = 128 MiB
    // per layer (the source doc's "~512 MiB" total is a 2x slip --
    // two layers of this spec are 256 MiB).
    let s = EngramSpec {
        hidden_size: 5_120,
        ngram_orders: vec![2, 3],
        heads_per_ngram: 4,
        slots_per_head: 262_144,
        head_dim: 32,
        seed: 7,
    };
    assert_eq!(s.parameter_count().unwrap(), 67_108_864);
    assert_eq!(s.bytes_for(DType::F16).unwrap(), 128 * 1024 * 1024);
    assert_eq!(s.bytes_for(DType::F32).unwrap(), 256 * 1024 * 1024);
}
