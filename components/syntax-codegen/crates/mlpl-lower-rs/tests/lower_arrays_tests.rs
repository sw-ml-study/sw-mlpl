//! String-match tests for the array / variable-binding / fncall
//! surface added in compile-to-rust step 003.

use mlpl_lower_rs::{LowerError, lower};
use mlpl_parser::{lex, parse};

fn lower_src(src: &str) -> Result<String, LowerError> {
    let tokens = lex(src).expect("lex ok");
    let stmts = parse(&tokens).expect("parse ok");
    lower(&stmts).map(|ts| ts.to_string())
}

#[test]
fn array_literal_scalars() {
    let s = lower_src("[1, 2, 3]").unwrap();
    assert!(s.contains("array_lit"), "{s}");
    assert!(s.contains("vec !"), "{s}");
    assert!(s.contains("1"), "{s}");
    assert!(s.contains("3"), "{s}");
}

#[test]
fn nested_array_literal() {
    // [[1, 2], [3, 4]] -- two layers of array_lit.
    let s = lower_src("[[1, 2], [3, 4]]").unwrap();
    assert!(s.matches("array_lit").count() >= 3, "{s}");
}

#[test]
fn assign_emits_let_binding() {
    let s = lower_src("x = 5").unwrap();
    assert!(s.contains("let x ="), "{s}");
    // Final yield returns the bound value.
    assert!(s.contains("x . clone"), "{s}");
}

#[test]
fn ident_reference_emits_clone() {
    let s = lower_src("x = 5\ny = x").unwrap();
    assert!(s.contains("let x ="), "{s}");
    assert!(s.contains("let y ="), "{s}");
    assert!(s.contains("x . clone"), "{s}");
}

#[test]
fn multi_stmt_program_yields_last() {
    // Two statements, last is not an Assign: its expression is the
    // block's yielded value (no trailing clone).
    let s = lower_src("x = 3\nx + 1").unwrap();
    assert!(s.contains("let x ="), "{s}");
    assert!(s.contains("apply_binop"), "{s}");
}

#[test]
fn fncall_iota() {
    let s = lower_src("iota(5)").unwrap();
    assert!(s.contains(":: mlpl_rt :: iota"), "{s}");
    assert!(s.contains("as usize"), "{s}");
}

#[test]
fn fncall_shape_rank_transpose() {
    for (src, expect) in [
        ("shape(iota(3))", ":: mlpl_rt :: shape"),
        ("rank(iota(3))", ":: mlpl_rt :: rank"),
        ("transpose(iota(6))", ":: mlpl_rt :: transpose"),
    ] {
        let s = lower_src(src).unwrap();
        assert!(s.contains(expect), "src={src}, out={s}");
    }
}

#[test]
fn fncall_reshape_threads_dims() {
    let s = lower_src("reshape(iota(6), [2, 3])").unwrap();
    assert!(s.contains(":: mlpl_rt :: reshape"), "{s}");
    assert!(s.contains("as usize"), "{s}");
}

#[test]
fn fncall_reduce_add_flat_and_axis() {
    let flat = lower_src("reduce_add(iota(4))").unwrap();
    assert!(flat.contains(":: mlpl_rt :: reduce_add"), "{flat}");
    // Per-axis reduce uses the distinct reduce_add_axis primitive.
    let by_axis = lower_src("reduce_add(reshape(iota(6), [2, 3]), 0)").unwrap();
    assert!(
        by_axis.contains(":: mlpl_rt :: reduce_add_axis"),
        "{by_axis}"
    );
}

#[test]
fn unsupported_construct_errors() {
    // repeat {} is out of scope for step 003.
    let r = lower_src("repeat 3 { x = 1 }");
    assert!(matches!(r, Err(LowerError::Unsupported(_))), "got {r:?}");
}

#[test]
fn unsupported_fncall_errors() {
    // softmax is not in the phase-1 primitive list.
    let r = lower_src("softmax(iota(3), 0)");
    assert!(
        matches!(r, Err(LowerError::Unsupported(ref m)) if m.contains("softmax")),
        "got {r:?}"
    );
}
