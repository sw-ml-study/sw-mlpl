//! `tests()` / `test_info()` / `annotations()` -- discovery
//! without execution over the registry and the general
//! annotation namespace.

use mlpl_eval::env_api::*;
use mlpl_eval::{Environment, Value};

fn eval_src(env: &mut Environment, src: &str) -> Result<Value, String> {
    let tokens = mlpl_parser::lex(src).map_err(|e| e.to_string())?;
    let stmts = mlpl_parser::parse(&tokens).map_err(|e| e.to_string())?;
    env.set_pending_source(Some(src.to_string()));
    let out = mlpl_eval::eval_program_value(&stmts, env).map_err(|e| e.to_string());
    env.set_pending_source(None);
    out
}

const SUITE: &str = r#"
@test
def u:alpha() { 1 }

@test {name: "beta case", tags: ["fast"], timeout_ms: 500}
def u:beta() { 2 }
"#;

#[test]
fn tests_lists_stable_names_in_source_order() {
    let mut env = Environment::new();
    eval_src(&mut env, SUITE).expect("suite");
    let out = eval_src(&mut env, "tests()").expect("tests()");
    let Value::StrList { items } = out else {
        panic!("expected string list, got {out:?}");
    };
    assert_eq!(items, vec!["alpha".to_string(), "beta case".to_string()]);
}

#[test]
fn test_info_returns_registry_row_with_callable_ref() {
    let mut env = Environment::new();
    eval_src(&mut env, SUITE).expect("suite");
    let out = eval_src(&mut env, "test_info(\"beta case\")").expect("info");
    let Value::Record { fields } = out else {
        panic!("expected record, got {out:?}");
    };
    assert_eq!(fields["name"], Value::Str("beta case".into()));
    assert!(matches!(&fields["fn"], Value::UserFnRef { name } if name == "u:beta"));
    assert!(matches!(&fields["tags"], Value::StrList { items } if items == &["fast".to_string()]));
    assert!(matches!(&fields["timeout_ms"], Value::Array(a) if a.data()[0] == 500.0));
    assert!(matches!(&fields["line"], Value::Array(a) if a.data()[0] == 6.0));
    assert_eq!(fields["skip"], Value::Str(String::new()));
    // discovered ref is invocable through call()
    let run = eval_src(&mut env, "call(test_info(\"beta case\").fn)").expect("call");
    assert!(matches!(run, Value::Array(a) if a.data()[0] == 2.0));
}

#[test]
fn test_info_unknown_name_is_loud_and_names_known_tests() {
    let mut env = Environment::new();
    eval_src(&mut env, SUITE).expect("suite");
    let err = eval_src(&mut env, "test_info(\"nope\")").expect_err("must fail");
    let msg = format!("{err:?}");
    assert!(msg.contains("no test named"), "got: {msg}");
    assert!(msg.contains("beta case"), "should list known tests: {msg}");
}

#[test]
fn annotations_exposes_general_namespace_with_bare_as_one() {
    let mut env = Environment::new();
    let src = r#"
@formula "H(p) = -sum(p * log(p))"
@doc {latex: "sum"}
@reviewed
def u:entropy(p) { 0 - sum(p * log(p)) }
"#;
    eval_src(&mut env, src).expect("defs");
    for name_arg in ["\"u:entropy\"", "\"entropy\""] {
        let out = eval_src(&mut env, &format!("annotations({name_arg})")).expect("annotations");
        let Value::Record { ref fields } = out else {
            panic!("expected record, got {out:?}");
        };
        assert_eq!(
            fields["formula"],
            Value::Str("H(p) = -sum(p * log(p))".into())
        );
        assert!(matches!(&fields["doc"], Value::Record { .. }));
        assert!(matches!(&fields["reviewed"], Value::Array(a) if a.data()[0] == 1.0));
    }
    let err = eval_src(&mut env, "annotations(\"u:missing\")").expect_err("unknown fn");
    assert!(format!("{err:?}").contains("undefined function"));
}
