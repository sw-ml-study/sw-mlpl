//! Tests for the `load` and `load_preloaded` builtins (Saga 12 step 001).
//!
//! Scope note: per the milestone doc, CSV headers were going to
//! become axis-1 labels. MLPL's label model is one label per AXIS
//! (not per column), so "axis 1 label = the whole column-name
//! string" is awkward. Dropped from this step; a later `load_labeled`
//! variant can attach per-column names through a different
//! mechanism (e.g. a parallel vector of column names).
//!
//! For this step, CSV load produces a plain DenseArray of numeric
//! cells with the header row stripped when it is non-numeric.

use std::path::PathBuf;

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn sandbox(files: &[(&str, &str)]) -> PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = base.join(format!("mlpl-load-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, body) in files {
        std::fs::write(dir.join(name), body).unwrap();
    }
    dir
}

fn eval_with_data_dir(src: &str, data_dir: Option<PathBuf>) -> Result<Value, mlpl_eval::EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    if let Some(d) = data_dir {
        env.set_data_dir(d);
    }
    eval_program_value(&stmts, &mut env)
}

// -- CSV --

#[test]
fn load_csv_with_header_row_strips_header() {
    let dir = sandbox(&[("nums.csv", "a,b,c\n1,2,3\n4,5,6\n")]);
    let v = eval_with_data_dir("load(\"nums.csv\")", Some(dir)).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.shape().dims(), &[2, 3]);
    assert_eq!(arr.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn load_csv_no_header_reads_all_rows_as_data() {
    let dir = sandbox(&[("pure.csv", "1,2\n3,4\n5,6\n")]);
    let v = eval_with_data_dir("load(\"pure.csv\")", Some(dir)).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.shape().dims(), &[3, 2]);
    assert_eq!(arr.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn load_csv_non_numeric_body_errors() {
    let dir = sandbox(&[("bad.csv", "1,2\nfoo,bar\n")]);
    let r = eval_with_data_dir("load(\"bad.csv\")", Some(dir));
    assert!(r.is_err(), "expected non-numeric-body error");
}

#[test]
fn load_csv_ragged_rows_error() {
    let dir = sandbox(&[("ragged.csv", "1,2\n3,4,5\n")]);
    let r = eval_with_data_dir("load(\"ragged.csv\")", Some(dir));
    assert!(r.is_err(), "expected ragged-row error");
}

#[test]
fn load_csv_single_column() {
    let dir = sandbox(&[("col.csv", "1\n2\n3\n")]);
    let v = eval_with_data_dir("load(\"col.csv\")", Some(dir)).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.shape().dims(), &[3, 1]);
    assert_eq!(arr.data(), &[1.0, 2.0, 3.0]);
}

// -- TXT --

#[test]
fn load_txt_returns_string() {
    let dir = sandbox(&[("greet.txt", "hello world\n")]);
    let v = eval_with_data_dir("load(\"greet.txt\")", Some(dir)).unwrap();
    assert_eq!(v, Value::Str("hello world\n".into()));
}

#[test]
fn load_txt_utf8_bytes_preserved() {
    let dir = sandbox(&[("utf8.txt", "h\u{00e9}llo")]);
    let v = eval_with_data_dir("load(\"utf8.txt\")", Some(dir)).unwrap();
    assert_eq!(v, Value::Str("h\u{00e9}llo".into()));
}

// -- Sandbox --

#[test]
fn load_absolute_path_errors() {
    let dir = sandbox(&[]);
    let r = eval_with_data_dir("load(\"/etc/passwd\")", Some(dir));
    let err = r.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("absolute") || msg.contains("sandbox"),
        "expected sandbox message, got: {msg}"
    );
}

#[test]
fn load_parent_traversal_errors() {
    let dir = sandbox(&[]);
    let r = eval_with_data_dir("load(\"../outside.csv\")", Some(dir));
    let err = r.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("sandbox") || msg.contains("outside"),
        "expected sandbox escape error, got: {msg}"
    );
}

#[test]
fn load_without_data_dir_errors_pointing_to_preloaded() {
    let r = eval_with_data_dir("load(\"anything.csv\")", None);
    let err = r.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("load_preloaded") || msg.contains("data-dir"),
        "expected pointer to load_preloaded or --data-dir, got: {msg}"
    );
}

// -- Preloaded --

#[test]
fn load_preloaded_returns_registered_corpus() {
    let v = eval_with_data_dir("load_preloaded(\"tiny_corpus\")", None).unwrap();
    match v {
        Value::Str(s) => assert!(!s.is_empty(), "tiny_corpus should be non-empty"),
        other => panic!("expected Value::Str, got {other:?}"),
    }
}

#[test]
fn load_preloaded_unknown_name_errors() {
    let r = eval_with_data_dir("load_preloaded(\"does_not_exist\")", None);
    let err = r.unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("does_not_exist"), "{msg}");
}
