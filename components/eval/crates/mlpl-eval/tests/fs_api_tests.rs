//! Sandboxed filesystem builtins (mlplunit's language-native-
//! runner gate, first half): fs_walk / read_text / write_text /
//! remove_path -- all Result-returning, all contained by the
//! environment's sandbox root, exact text preserved, lexical
//! order, paths relative to the root so results feed straight
//! back into the other calls.

use mlpl_eval::{Environment, Value};

fn eval_value(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string())
}

fn sandbox(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mlpl-fs-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("nested")).unwrap();
    std::fs::write(dir.join("test_alpha.mlpl"), "1 + 1\n").unwrap();
    std::fs::write(dir.join("notes.txt"), "not a test").unwrap();
    std::fs::write(dir.join("nested/test_beta.mlpl"), "2 + 2\n").unwrap();
    dir
}

fn env_with(dir: &std::path::Path) -> Environment {
    let mut env = Environment::new();
    env.fs_root = Some(dir.to_path_buf());
    env
}

#[test]
fn walk_filters_recursively_in_lexical_order() {
    let dir = sandbox("walk");
    let mut env = env_with(&dir);
    let v = eval_value(
        &mut env,
        "unwrap(fs_walk(\".\", {recursive: 1, kind: \"file\", pattern: \"test_*.mlpl\"}))",
    )
    .unwrap();
    let Value::StrList { items } = v else {
        panic!("expected string list: {v:?}")
    };
    assert_eq!(items, vec!["nested/test_beta.mlpl", "test_alpha.mlpl"]);
    // Non-recursive walk stays at the top level.
    let v = eval_value(
        &mut env,
        "unwrap(fs_walk(\".\", {recursive: 0, kind: \"file\", pattern: \"*.mlpl\"}))",
    )
    .unwrap();
    assert!(matches!(&v, Value::StrList { items } if items == &["test_alpha.mlpl"]));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn text_round_trip_is_exact_including_unicode() {
    let dir = sandbox("round");
    let mut env = env_with(&dir);
    eval_value(
        &mut env,
        "unwrap(write_text(\"copy.mlpl\", \"# ⍳ glyphs and line two\"))",
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let v = eval_value(&mut env, "unwrap(read_text(\"copy.mlpl\"))").unwrap();
    assert!(
        matches!(&v, Value::Str(s) if s == "# ⍳ glyphs and line two"),
        "{v:?}"
    );
    let v = eval_value(&mut env, "unwrap(remove_path(\"copy.mlpl\"))").unwrap();
    assert!(matches!(&v, Value::Array(a) if a.data()[0] == 1.0));
    let v = eval_value(&mut env, "is_err(read_text(\"copy.mlpl\"))").unwrap();
    assert!(matches!(&v, Value::Array(a) if a.data()[0] == 1.0));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn traversal_outside_the_sandbox_is_an_err_value() {
    let dir = sandbox("jail");
    let mut env = env_with(&dir);
    for src in [
        "is_err(read_text(\"../outside.txt\"))",
        "is_err(write_text(\"../outside.txt\", \"x\"))",
        "is_err(remove_path(\"../outside.txt\"))",
        "is_err(fs_walk(\"..\", {recursive: 0, kind: \"file\", pattern: \"*\"}))",
    ] {
        let v = eval_value(&mut env, src).unwrap_or_else(|e| panic!("{src}: {e}"));
        assert!(
            matches!(&v, Value::Array(a) if a.data()[0] == 1.0),
            "{src} must be err"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_sandbox_root_is_a_plain_err() {
    let mut env = Environment::new();
    let v = eval_value(&mut env, "err_message(read_text(\"x\"))").unwrap();
    assert!(
        matches!(&v, Value::Str(s) if s.contains("sandbox")),
        "{v:?}"
    );
}

#[test]
fn walk_results_feed_read_text_directly() {
    let dir = sandbox("feed");
    let mut env = env_with(&dir);
    let v = eval_value(
        &mut env,
        "files = unwrap(fs_walk(\".\", {recursive: 1, kind: \"file\", pattern: \"test_*.mlpl\"}))\n\
         unwrap(read_text(unwrap(list_get(files, 1))))",
    )
    .unwrap();
    assert!(matches!(&v, Value::Str(s) if s == "1 + 1\n"), "{v:?}");
    std::fs::remove_dir_all(&dir).ok();
}
