//! Saga R1 step 002: `Value::DeviceTensor` strict-
//! fault tests. The new variant must error
//! actionable when a CPU op tries to consume a
//! peer-resident tensor without explicit
//! `to_device('cpu', ...)`.

use mlpl_eval::{EvalError, Value};

fn device_tensor() -> Value {
    Value::DeviceTensor {
        peer: "http://mac.local:6465".into(),
        handle: "abc123".into(),
        shape: vec![3, 4],
        device: "mlx".into(),
    }
}

#[test]
fn into_array_on_device_tensor_returns_actionable_fault() {
    let v = device_tensor();
    let err = v.into_array().expect_err("expected DeviceTensorFault");
    match err {
        EvalError::DeviceTensorFault { peer, device } => {
            assert_eq!(peer, "http://mac.local:6465");
            assert_eq!(device, "mlx");
        }
        other => panic!("expected DeviceTensorFault, got {other:?}"),
    }
}

#[test]
fn as_array_on_device_tensor_returns_actionable_fault() {
    let v = device_tensor();
    let err = v.as_array().expect_err("expected DeviceTensorFault");
    let msg = format!("{err}");
    assert!(
        msg.contains("tensor lives on") && msg.contains("to_device('cpu'"),
        "fault message should be actionable: {msg}"
    );
    assert!(msg.contains("mlx"), "msg should mention the device: {msg}");
    assert!(
        msg.contains("mac.local:6465"),
        "msg should mention the peer: {msg}"
    );
}

#[test]
fn display_prints_peer_and_device_and_shape() {
    let v = device_tensor();
    let s = format!("{v}");
    assert!(s.contains("mlx"), "Display should include device: {s}");
    assert!(
        s.contains("mac.local:6465"),
        "Display should include peer: {s}"
    );
    assert!(
        s.contains("3") && s.contains("4"),
        "Display should include shape: {s}"
    );
}
