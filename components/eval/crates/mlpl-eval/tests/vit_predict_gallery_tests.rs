//! `demos/vit_predict_gallery.mlpl` + predict_batch builtin
//! integration tests. Saga 29 step 011.

use std::fs;
use std::path::PathBuf;

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, Value, eval_program, eval_program_value};
use mlpl_parser::{lex, parse};

fn arr(dims: Vec<usize>, data: Vec<f64>) -> DenseArray {
    DenseArray::new(Shape::new(dims), data).unwrap()
}

fn run(src: &str, env: &mut Environment) -> DenseArray {
    eval_program(&parse(&lex(src).unwrap()).unwrap(), env).unwrap()
}

#[test]
fn predict_batch_returns_argmax_over_last_axis() {
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("mdl = chain(linear(4, 3, 7), softmax_layer())").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    env.set(
        "X".into(),
        arr(
            vec![3, 4],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ),
    );
    let y = run("predict_batch(mdl, X)", &mut env);
    assert_eq!(y.shape().dims(), &[3]);
    // Values are integer class indices in [0, num_classes).
    for &v in y.data() {
        assert!((0.0..3.0).contains(&v) && v.fract() == 0.0);
    }
}

#[test]
fn predict_batch_arity_error() {
    let mut env = Environment::new();
    eval_program(
        &parse(&lex("mdl = linear(2, 2, 0)").unwrap()).unwrap(),
        &mut env,
    )
    .unwrap();
    let res = eval_program_value(
        &parse(&lex("predict_batch(mdl)").unwrap()).unwrap(),
        &mut env,
    );
    assert!(res.is_err());
}

#[test]
fn predict_batch_unknown_model_error() {
    let mut env = Environment::new();
    env.set("X".into(), arr(vec![1, 2], vec![0.0, 0.0]));
    let res = eval_program_value(
        &parse(&lex("predict_batch(nope, X)").unwrap()).unwrap(),
        &mut env,
    );
    assert!(res.is_err());
}

#[test]
#[ignore = "trained ViT + predict + gallery: 50 adam steps on B=16 (~2-3 minutes on CPU)"]
fn vit_predict_gallery_demo_runs_end_to_end() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../demos/vit_predict_gallery.mlpl");
    let src = fs::read_to_string(&path).expect("read demo file");
    let tokens = lex(&src).expect("lex");
    let stmts = parse(&tokens).expect("parse");
    let mut env = Environment::new();
    let v = eval_program_value(&stmts, &mut env).expect("eval");
    // Final value is the svg(...) call returning a Value::Str.
    match v {
        Value::Str(s) => {
            assert!(s.contains("<svg"), "expected svg output, got {s:.80}");
            // Overlay text elements present.
            assert!(
                s.contains("<text"),
                "gallery should include overlay <text> elements"
            );
        }
        other => panic!("expected svg string, got {other:?}"),
    }
}
