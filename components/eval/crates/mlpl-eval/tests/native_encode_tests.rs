//! to_native(value) -- the typed-native binary encoder. Produces
//! an ok(byte array) with the MLPB header; non-data kinds err;
//! output is deterministic; bytes compose with write_bytes.

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

#[test]
fn encodes_with_the_mlpb_header() {
    let mut env = Environment::new();
    eval_value(&mut env, "b = unwrap(to_native(42))").unwrap();
    // magic "MLPB" = 77 76 80 66, then version 1
    assert_eq!(scalar(&mut env, "take(b, 0, 0)"), 77.0);
    assert_eq!(scalar(&mut env, "take(b, 0, 1)"), 76.0);
    assert_eq!(scalar(&mut env, "take(b, 0, 2)"), 80.0);
    assert_eq!(scalar(&mut env, "take(b, 0, 3)"), 66.0);
    assert_eq!(scalar(&mut env, "take(b, 0, 4)"), 1.0);
}

#[test]
fn is_ok_for_data_kinds() {
    let mut env = Environment::new();
    assert_eq!(scalar(&mut env, "is_ok(to_native(42))"), 1.0);
    assert_eq!(scalar(&mut env, "is_ok(to_native([1, 2, 3]))"), 1.0);
    assert_eq!(scalar(&mut env, "is_ok(to_native(\"hi\"))"), 1.0);
    assert_eq!(scalar(&mut env, "is_ok(to_native({a: 1, b: 2}))"), 1.0);
    assert_eq!(scalar(&mut env, "is_ok(to_native(ok(5)))"), 1.0);
    assert_eq!(
        scalar(&mut env, "is_ok(to_native(reshape(range(4), [2, 2])))"),
        1.0
    );
}

#[test]
fn non_data_kinds_are_err() {
    let mut env = Environment::new();
    assert_eq!(
        scalar(&mut env, "is_ok(to_native(chain(linear(2, 2, 0))))"),
        0.0
    );
}

#[test]
fn output_is_deterministic_and_sorted() {
    let mut env = Environment::new();
    // insertion order does not matter -- records encode sorted
    assert_eq!(
        scalar(
            &mut env,
            "equal(unwrap(to_native({a: 1, b: 2})), unwrap(to_native({b: 2, a: 1})))"
        ),
        1.0
    );
}

#[test]
fn bytes_compose_with_write_bytes() {
    let dir = std::env::temp_dir().join(format!("mlpl-native-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut env = Environment::new();
    env.fs_root = Some(dir.clone());
    assert_eq!(
        scalar(
            &mut env,
            "unwrap(write_bytes(\"v.bin\", unwrap(to_native({a: 1, b: 2}))))"
        ),
        1.0
    );
    std::fs::remove_dir_all(&dir).ok();
}
