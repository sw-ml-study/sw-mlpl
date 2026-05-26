Step 001: Add eval_with_values to WasmSession.

Add a new method to WasmSession (crates/mlpl-wasm/src/lib.rs) that returns both the display string AND the raw DenseArray data:

pub fn eval_with_values(&self, input: &str) -> (String, Option<Vec<f64>>, Vec<usize>)

Returns (display_string, optional_flat_values, shape). For non-array results (strings, errors, models), values is None. For arrays, values is Some(data.to_vec()) and shape is the array shape.

This avoids parsing display strings back into numbers. The existing eval() stays unchanged for backwards compat.

Test: unit test that eval_with_values('iota(5)') returns correct values + shape.