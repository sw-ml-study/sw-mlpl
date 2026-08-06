//! Total structural equality. NEVER errors: mismatched kinds
//! compare unequal. Numbers use IEEE equality except that NaN
//! equals NaN (assertions on NaN-bearing arrays stay honest);
//! 0.0 and -0.0 compare equal per IEEE. Arrays compare shape,
//! axis labels, and elements. Device tensors always compare
//! unequal (fetch to the host first).

use mlpl_array::DenseArray;
use mlpl_eval_types::Value;

/// Structural equality over any two values.
// match_same_arms: several kinds compare with `x == y` on
// different binding types; merging the arms would obscure the
// kind-by-kind contract this function IS.
#[allow(clippy::match_same_arms)]
#[must_use]
pub fn value_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => arrays_equal(x, y),
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::StrList { items: x }, Value::StrList { items: y }) => x == y,
        (Value::Record { .. }, Value::Record { .. })
        | (Value::Result { .. }, Value::Result { .. }) => compound_equal(a, b),
        (Value::Model(x), Value::Model(y)) => x == y,
        (Value::Tokenizer(x), Value::Tokenizer(y)) => x == y,
        (Value::BuiltinRef { name: x }, Value::BuiltinRef { name: y }) => x == y,
        _ => false,
    }
}

/// `Record` and `Result` recursion, split out for the LOC budget.
fn compound_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Record { fields: x }, Value::Record { fields: y }) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|((ka, va), (kb, vb))| ka == kb && value_equal(va, vb))
        }
        (
            Value::Result {
                ok: oa,
                payload: pa,
            },
            Value::Result {
                ok: ob,
                payload: pb,
            },
        ) => oa == ob && value_equal(pa, pb),
        _ => false,
    }
}

// float_cmp: EXACT equality is the documented semantics -- this
// is an assertion primitive, not numeric tolerance.
#[allow(clippy::float_cmp)]
fn arrays_equal(x: &DenseArray, y: &DenseArray) -> bool {
    x.shape().dims() == y.shape().dims()
        && x.labels() == y.labels()
        && x.data()
            .iter()
            .zip(y.data())
            .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
}
