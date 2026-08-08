//! Wrap a codec's `Result<Value, String>` into the ok/err MLPL
//! Result value the JSON/TOML builtins return (ok payload is the
//! value, err payload the message string).

use mlpl_eval_types::Value;

pub(crate) fn ok_or_err(r: Result<Value, String>) -> Value {
    match r {
        Ok(v) => Value::Result {
            ok: true,
            payload: Box::new(v),
        },
        Err(m) => Value::Result {
            ok: false,
            payload: Box::new(Value::Str(m)),
        },
    }
}
