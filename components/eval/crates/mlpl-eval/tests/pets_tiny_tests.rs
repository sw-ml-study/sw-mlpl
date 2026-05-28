//! `load_preloaded("pets_tiny")` tests. Saga 29 step 003.
//!
//! These tests run regardless of the `image-io` feature
//! because the fixture (`crates/mlpl-eval/data/pets_tiny.bin`)
//! is pre-decoded u8 RGB shipped via `include_bytes!`. The
//! WASM REPL exercises the same code path.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Result<Value, mlpl_eval::EvalError> {
    let tokens = lex(src).unwrap();
    let stmts = parse(&tokens).unwrap();
    let mut env = Environment::new();
    eval_program_value(&stmts, &mut env)
}

fn load_record() -> Value {
    eval("load_preloaded(\"pets_tiny\")").expect("load_preloaded(pets_tiny)")
}

#[test]
fn pets_tiny_returns_a_record() {
    match load_record() {
        Value::Record { fields } => {
            let mut keys: Vec<&str> = fields.keys().map(String::as_str).collect();
            keys.sort();
            assert_eq!(keys, vec!["X", "Y", "names"]);
        }
        other => panic!("expected Record, got {other:?}"),
    }
}

#[test]
fn pets_tiny_x_has_shape_200_3_64_64() {
    let r = load_record();
    let Value::Record { fields } = r else {
        panic!("expected Record");
    };
    let Some(Value::Array(x)) = fields.get("X") else {
        panic!("X should be Array");
    };
    assert_eq!(x.shape().dims(), &[200, 3, 64, 64]);
}

#[test]
fn pets_tiny_x_has_batch_channel_y_x_labels() {
    let r = load_record();
    let Value::Record { fields } = r else {
        panic!("expected Record");
    };
    let Some(Value::Array(x)) = fields.get("X") else {
        panic!("X should be Array");
    };
    let labels = x.labels().expect("X must carry axis labels");
    let names: Vec<&str> = labels.iter().map(|l| l.as_deref().unwrap_or("")).collect();
    assert_eq!(names, vec!["batch", "channel", "y", "x"]);
}

#[test]
fn pets_tiny_x_values_are_in_minus_one_to_one() {
    let r = load_record();
    let Value::Record { fields } = r else {
        panic!("expected Record");
    };
    let Some(Value::Array(x)) = fields.get("X") else {
        panic!("X should be Array");
    };
    for &v in x.data().iter().take(10_000) {
        assert!((-1.0..=1.0).contains(&v), "pixel out of [-1, 1]: {v}");
    }
}

#[test]
fn pets_tiny_y_has_200_entries_split_100_cats_100_dogs() {
    let r = load_record();
    let Value::Record { fields } = r else {
        panic!("expected Record");
    };
    let Some(Value::Array(y)) = fields.get("Y") else {
        panic!("Y should be Array");
    };
    assert_eq!(y.shape().dims(), &[200]);
    let cats = y.data().iter().filter(|&&v| v == 0.0).count();
    let dogs = y.data().iter().filter(|&&v| v == 1.0).count();
    assert_eq!(cats, 100, "expected 100 cats (label 0), got {cats}");
    assert_eq!(dogs, 100, "expected 100 dogs (label 1), got {dogs}");
}

#[test]
fn pets_tiny_y_has_batch_label() {
    let r = load_record();
    let Value::Record { fields } = r else {
        panic!("expected Record");
    };
    let Some(Value::Array(y)) = fields.get("Y") else {
        panic!("Y should be Array");
    };
    let labels = y.labels().expect("Y must carry axis labels");
    let names: Vec<&str> = labels.iter().map(|l| l.as_deref().unwrap_or("")).collect();
    assert_eq!(names, vec!["batch"]);
}

#[test]
fn pets_tiny_names_is_a_strlist_of_200_basenames() {
    let r = load_record();
    let Value::Record { fields } = r else {
        panic!("expected Record");
    };
    let Some(Value::StrList { items }) = fields.get("names") else {
        panic!("names should be StrList");
    };
    assert_eq!(
        items.len(),
        200,
        "expected 200 filenames, got {}",
        items.len()
    );
    // First entry is index 0 (a cat) so name should start
    // with an uppercase letter (Oxford-IIIT Pet convention).
    let first = &items[0];
    assert!(
        first.chars().next().unwrap().is_ascii_uppercase(),
        "first name should be a cat (uppercase prefix), got {first}"
    );
    // 101st entry is the first dog -> lowercase prefix.
    let first_dog = &items[100];
    assert!(
        first_dog.chars().next().unwrap().is_ascii_lowercase(),
        "first dog name should have lowercase prefix, got {first_dog}"
    );
    // Every name should end in .jpg or .jpeg.
    for name in items {
        let lower = name.to_ascii_lowercase();
        assert!(
            lower.ends_with(".jpg") || lower.ends_with(".jpeg"),
            "name without jpg/jpeg extension: {name}"
        );
    }
}

#[test]
fn pets_tiny_x_y_names_are_consistent_across_index() {
    // Cat at index 0 has Y=0 and uppercase-prefix name.
    // Dog at index 100 has Y=1 and lowercase-prefix name.
    let r = load_record();
    let Value::Record { fields } = r else {
        panic!("expected Record");
    };
    let y = match fields.get("Y") {
        Some(Value::Array(a)) => a,
        _ => panic!("Y missing"),
    };
    let names = match fields.get("names") {
        Some(Value::StrList { items }) => items,
        _ => panic!("names missing"),
    };
    assert_eq!(y.data()[0], 0.0, "index 0 should be cat");
    assert_eq!(y.data()[100], 1.0, "index 100 should be dog");
    assert!(names[0].chars().next().unwrap().is_ascii_uppercase());
    assert!(names[100].chars().next().unwrap().is_ascii_lowercase());
}

#[test]
fn pets_tiny_unknown_preloaded_still_errors_cleanly() {
    // Regression: adding pets_tiny didn't break the generic
    // "unknown preloaded corpus" path.
    let err = eval("load_preloaded(\"definitely_not_a_corpus\")").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unknown preloaded corpus"),
        "error should mention unknown corpus, got: {msg}"
    );
}

#[test]
fn pets_tiny_record_field_access_works() {
    // The whole point of step 001 (Value::Record) +
    // step 002 (Value::StrList) was making this expressible.
    let v = eval(
        "r = load_preloaded(\"pets_tiny\")\n\
         list_len(r.names)",
    )
    .expect("field access + list_len chain");
    match v {
        Value::Array(a) => {
            assert_eq!(a.data(), &[200.0]);
        }
        other => panic!("expected scalar Array of 200, got {other:?}"),
    }
}
