//! Pins the transducers literate document (examples/literate/
//! transducers.org) and its tangled program: the tangled `.mlpl` is the
//! org's source blocks in order, every `def` carries a docstring, and the
//! program computes the results the prose and baked RESULTS claim.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

const ORG: &str = include_str!("../../../../../examples/literate/transducers.org");
const TANGLED: &str = include_str!("../../../../../examples/literate/transducers.mlpl");

/// The org's `mlpl` blocks, concatenated in order, blank lines dropped.
fn org_program() -> Vec<&'static str> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in ORG.lines() {
        let t = line.trim_start();
        if t.starts_with("#+begin_src mlpl") {
            inside = true;
        } else if t.starts_with("#+end_src") {
            inside = false;
        } else if inside && !t.is_empty() {
            out.push(line);
        }
    }
    out
}

fn eval(env: &mut Environment, src: &str) -> Value {
    eval_program_value(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

fn nums(v: Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.data().to_vec(),
        other => panic!("expected an array, got {other:?}"),
    }
}

#[test]
fn tangled_program_matches_the_org_blocks() {
    let tangled: Vec<&str> = TANGLED.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(tangled, org_program(), "re-tangle transducers.org");
}

#[test]
fn every_def_has_a_docstring() {
    let lines: Vec<&str> = TANGLED.lines().collect();
    for (i, l) in lines.iter().enumerate() {
        if l.trim_start().starts_with("def ") {
            let next = lines.get(i + 1).map_or("", |n| n.trim_start());
            assert!(next.starts_with('"'), "no docstring: {l}");
        }
    }
}

#[test]
fn the_document_computes_what_it_claims() {
    let mut env = Environment::new();
    eval(&mut env, TANGLED);
    let mut num = |src: &str| nums(eval(&mut env, src));
    assert_eq!(
        num("u:transduce(odd_squares, sum_rf, 0, iota(10))"),
        vec![165.0]
    );
    assert_eq!(
        num("u:transduce(first3, sum_rf, 0, iota(1000000))"),
        vec![35.0]
    );
    assert_eq!(
        num("u:transduce(odd_squares, u:reducer(:u:conj), [], xs)"),
        vec![1.0, 9.0, 25.0, 49.0, 81.0]
    );
    assert_eq!(num("u:transduce(long_chars, sum_rf, 0, words)"), vec![21.0]);
    assert_eq!(num("u:transduce(:u:catting, sum_rf, 0, grid)"), vec![15.0]);
    assert_eq!(num("streamed"), vec![1_333_333_330_000.0]);
    assert_eq!(num("hist"), vec![2.0, 0.0, 2.0, 2.0]);
    assert_eq!(num("stats.n"), vec![6.0]);
    assert!((num("stats.mean")[0] - 3.2 / 6.0).abs() < 1e-12);
    assert!(num("reduce_add(ok * readings)")[0].is_nan());
    assert_eq!(num("reduce_add(compress(ok, readings))"), vec![160.0]);
    match eval(&mut env, "joined") {
        Value::Str(s) => assert_eq!(s, "1, 9, 25, 49, 81"),
        other => panic!("{other:?}"),
    }
}
