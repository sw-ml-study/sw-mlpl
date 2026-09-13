//! moe-microscope finding F13: attention_weights must find an Attention layer
//! nested inside residual(chain(...)), not only at a chain's top level.

use mlpl_eval::{Environment, Value, eval_program_value};
use mlpl_parser::{lex, parse};

fn eval(src: &str) -> Value {
    eval_program_value(&parse(&lex(src).unwrap()).unwrap(), &mut Environment::new()).unwrap()
}

#[test]
fn attention_weights_finds_layer_inside_residual_chain() {
    // attention nested two levels deep: chain(residual(chain(rms, attn)), linear).
    let v = eval(
        "body = chain(residual(chain(rms_norm(8), causal_attention(8, 1, 1))), linear(8, 20, 2))\n\
         h = randn(1, [5, 8])\n\
         attention_weights(body, h)",
    );
    // A [5, 5] per-position attention matrix (single head, seq 5).
    let arr = match v {
        Value::Array(a) => a,
        other => panic!("expected an array, got {other:?}"),
    };
    assert_eq!(arr.shape().dims(), &[5, 5]);
    // Each row of a softmax attention matrix sums to 1.
    for row in arr.data().chunks(5) {
        let s: f64 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "row sums to 1: {row:?}");
    }
}
