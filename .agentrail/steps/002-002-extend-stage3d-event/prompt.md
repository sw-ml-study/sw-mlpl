Step 002: Extend Stage3dEvent with values.

Add optional values field to Stage3dEvent and ShapeInfo:

pub values: Option<Vec<f64>>   // actual array elements
pub summary: Option<ArraySummary>  // for large arrays

ArraySummary: min, max, mean, std, 16-bin histogram.

For arrays <=1000 elements, send all values. For larger arrays, send summary only. Update the emission in handlers_submit to call eval_with_values when available (wasm32 target) and populate the values field.

Pages rebuild required.