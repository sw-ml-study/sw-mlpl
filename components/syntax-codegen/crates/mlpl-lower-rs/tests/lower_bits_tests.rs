//! Bit-operation lowering (compiler-byte-io step 1). The pure bit-op
//! family lowers to a single `bit_try_call(name, vec![args])` runtime
//! dispatch (re-exported from `mlpl-runtime-bits`); invalid-domain
//! inputs are hard errors (the generated `.unwrap()` panics), matching
//! the interpreter's `RuntimeError`.

use mlpl_lower_rs::lower;
use mlpl_parser::{lex, parse};

fn lower_src(src: &str) -> String {
    let tokens = lex(src).expect("lex ok");
    let stmts = parse(&tokens).expect("parse ok");
    lower(&stmts).expect("lower ok").to_string()
}

#[test]
fn band_lowers_to_bit_try_call() {
    let s = lower_src("band(12, 10)");
    assert!(s.contains("bit_try_call"), "{s}");
    assert!(s.contains("\"band\""), "{s}");
    // Domain errors are hard errors: the dispatch is unwrapped.
    assert!(s.contains(". unwrap ()"), "{s}");
}

#[test]
fn every_bit_op_lowers_by_name() {
    for (src, name) in [
        ("bor(1, 2)", "bor"),
        ("bxor(1, 2)", "bxor"),
        ("bnot(10, 8)", "bnot"),
        ("popcount(255)", "popcount"),
        ("shl(15, 4, 8)", "shl"),
        ("shr(240, 4)", "shr"),
        ("bmask(255, 4)", "bmask"),
        ("bits(165, 8)", "bits"),
        ("from_bits([1, 0, 1])", "from_bits"),
    ] {
        let s = lower_src(src);
        assert!(s.contains("bit_try_call"), "{name}: {s}");
        assert!(s.contains(&format!("\"{name}\"")), "{name}: {s}");
    }
}

#[test]
fn bit_op_args_are_threaded_in_order() {
    // shl(x, k, width) passes all three args into the dispatch vec.
    let s = lower_src("shl(15, 4, 8)");
    assert!(s.contains("vec !"), "{s}");
    assert!(s.contains("15"), "{s}");
    assert!(s.contains('4'), "{s}");
    assert!(s.contains('8'), "{s}");
}
