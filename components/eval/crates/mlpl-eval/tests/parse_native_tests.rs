//! parse_native(bytes[, limits]) -- the typed-native binary
//! decoder. Round-trips every data kind losslessly with to_native;
//! validates header/version/length; enforces decode budgets;
//! malformed/truncated/over-budget input is an err Result.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn scalar(env: &mut Environment, src: &str) -> f64 {
    match eval_value(env, src).unwrap_or_else(|e| panic!("{src}: {e}")) {
        Value::Array(a) => a.data()[0],
        other => panic!("expected scalar from {src}, got {other:?}"),
    }
}

/// A round-trip equality check: v == parse_native(to_native(v)).
fn round_trips(env: &mut Environment, v: &str) -> f64 {
    eval_value(env, &format!("val = {v}")).unwrap();
    scalar(
        env,
        "equal(val, unwrap(parse_native(unwrap(to_native(val)))))",
    )
}

#[test]
fn round_trips_every_data_kind() {
    let mut env = Environment::new();
    for v in [
        "42",
        "3.5",
        "[1, 2, 3]",
        "[\"a\", \"b\"]",
        "\"hello world\"",
        "{a: 1, b: \"x\", c: [1, 2]}",
        "reshape(range(12), [2, 2, 3])", // rank-3, exact shape
        "ok(7)",
        "err(\"boom\")",
        "ok({a: [1, 2], b: err(\"nested\")})", // nested results + records
    ] {
        assert_eq!(round_trips(&mut env, v), 1.0, "round trip failed for {v}");
    }
}

#[test]
fn higher_rank_shape_is_preserved() {
    let mut env = Environment::new();
    eval_value(&mut env, "m = reshape(range(4), [2, 2])").unwrap();
    eval_value(
        &mut env,
        "back = unwrap(parse_native(unwrap(to_native(m))))",
    )
    .unwrap();
    assert_eq!(scalar(&mut env, "rank(back)"), 2.0);
    assert_eq!(scalar(&mut env, "equal(shape(back), [2, 2])"), 1.0);
}

#[test]
fn corrupt_and_truncated_input_is_err() {
    let mut env = Environment::new();
    // not an MLPB buffer
    assert_eq!(
        scalar(&mut env, "is_ok(parse_native([1, 2, 3, 4, 5]))"),
        0.0
    );
    // truncated: drop the last byte of a valid buffer
    eval_value(&mut env, "b = unwrap(to_native({a: 1, b: 2}))").unwrap();
    assert_eq!(
        scalar(
            &mut env,
            "is_ok(parse_native(compress(lt(range(tally(b)), tally(b) - 1), b)))"
        ),
        0.0
    );
}

#[test]
fn over_budget_input_is_err() {
    let mut env = Environment::new();
    eval_value(&mut env, "b = unwrap(to_native([1, 2, 3, 4, 5, 6, 7, 8]))").unwrap();
    // max_bytes below the buffer size
    assert_eq!(
        scalar(&mut env, "is_ok(parse_native(b, {max_bytes: 4}))"),
        0.0
    );
    // max_elements below the element count
    assert_eq!(
        scalar(&mut env, "is_ok(parse_native(b, {max_elements: 3}))"),
        0.0
    );
    // generous limits succeed
    assert_eq!(
        scalar(&mut env, "is_ok(parse_native(b, {max_bytes: 100000}))"),
        1.0
    );
}

#[test]
fn file_round_trip_composes_with_byte_io() {
    let dir = std::env::temp_dir().join(format!("mlpl-native-rt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut env = Environment::new();
    env.fs_root = Some(dir.clone());
    eval_value(&mut env, "v = {name: \"model\", dims: [2, 3], ok: ok(1)}").unwrap();
    eval_value(&mut env, "write_atomic(\"v.bin\", unwrap(to_native(v)))").unwrap();
    assert_eq!(
        scalar(
            &mut env,
            "equal(v, unwrap(parse_native(unwrap(read_bytes(\"v.bin\")))))"
        ),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}
