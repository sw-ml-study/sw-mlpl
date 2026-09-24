//! `save_model` files written before bias-free `linear` and the `rms_norm`
//! eps option still load: a string bias reads back as `Some`, a missing eps
//! takes the default.

use mlpl_eval_core::model::{DEFAULT_RMS_EPS, ModelSpec};

#[test]
fn old_linear_and_rms_norm_json_still_loads() {
    let old =
        r#"{"Chain":[{"Linear":{"w":"__linear_W_0","b":"__linear_b_0"}},{"RmsNorm":{"dim":4}}]}"#;
    let spec: ModelSpec = serde_json::from_str(old).expect("old format loads");
    let ModelSpec::Chain(c) = &spec else {
        panic!("{spec:?}")
    };
    assert_eq!(
        c[0],
        ModelSpec::Linear {
            w: "__linear_W_0".into(),
            b: Some("__linear_b_0".into())
        }
    );
    assert_eq!(
        c[1],
        ModelSpec::RmsNorm {
            dim: 4,
            eps: DEFAULT_RMS_EPS
        }
    );
    assert_eq!(spec.params(), vec!["__linear_W_0", "__linear_b_0"]);
}
