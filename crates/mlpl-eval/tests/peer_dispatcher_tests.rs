//! Saga R1 step 004: evaluator hooks for remote
//! peer dispatch. These tests use a fake dispatcher
//! so the eval crate can prove block forwarding and
//! explicit CPU materialization without depending on
//! the HTTP service crate.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use mlpl_array::{DenseArray, Shape};
use mlpl_eval::{Environment, EvalError, PeerDispatcher, Value, eval_program_value};
use mlpl_parser::{lex, parse};

#[derive(Debug, Default)]
struct FakeDispatcher {
    calls: Mutex<Vec<(String, String, Vec<String>)>>,
}

impl PeerDispatcher for FakeDispatcher {
    fn dispatch_block(
        &self,
        device: &str,
        source: &str,
        bindings: HashMap<String, DenseArray>,
    ) -> Option<Result<Value, EvalError>> {
        let mut names: Vec<String> = bindings.keys().cloned().collect();
        names.sort();
        self.calls
            .lock()
            .unwrap()
            .push((device.to_string(), source.to_string(), names));
        Some(Ok(Value::DeviceTensor {
            peer: "http://localhost:6465".into(),
            handle: "h1".into(),
            shape: vec![3],
            device: device.into(),
        }))
    }

    fn fetch_tensor(&self, peer: &str, handle: &str) -> Result<DenseArray, EvalError> {
        assert_eq!(peer, "http://localhost:6465");
        assert_eq!(handle, "h1");
        DenseArray::new(Shape::new(vec![3]), vec![0.0, 1.0, 2.0]).map_err(EvalError::from)
    }
}

fn eval(src: &str, env: &mut Environment) -> Value {
    let stmts = parse(&lex(src).expect("lex")).expect("parse");
    eval_program_value(&stmts, env).expect("eval")
}

#[test]
fn device_block_uses_peer_dispatcher_and_collects_array_bindings() {
    let dispatcher = Arc::new(FakeDispatcher::default());
    let mut env = Environment::new();
    env.set_peer_dispatcher(dispatcher.clone());
    eval("x = iota(3)", &mut env);

    let v = eval("device(\"mlx\") { x + 1 }", &mut env);
    match v {
        Value::DeviceTensor { device, shape, .. } => {
            assert_eq!(device, "mlx");
            assert_eq!(shape, vec![3]);
        }
        other => panic!("expected DeviceTensor, got {other:?}"),
    }
    let calls = dispatcher.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "mlx");
    assert!(calls[0].1.contains("x"));
    assert_eq!(calls[0].2, vec!["x"]);
}

#[test]
fn assigned_device_tensor_faults_until_explicit_cpu_fetch() {
    let dispatcher = Arc::new(FakeDispatcher::default());
    let mut env = Environment::new();
    env.set_peer_dispatcher(dispatcher);
    eval("x = device(\"mlx\") { iota(3) }", &mut env);

    let err_src = "x + 1";
    let stmts = parse(&lex(err_src).expect("lex")).expect("parse");
    let err = eval_program_value(&stmts, &mut env).expect_err("strict fault");
    assert!(format!("{err}").contains("to_device('cpu', x)"));

    let fetched = eval("to_device(\"cpu\", x)", &mut env);
    let arr = fetched.into_array().expect("array");
    assert_eq!(arr.data(), &[0.0, 1.0, 2.0]);
    assert_eq!(env.tensor_device("x"), "cpu");
}
