use mlpl_web_viz_ir::VizNode;
use serde::Serialize;
use wasm_bindgen::JsCast;

#[derive(Serialize)]
pub struct ShapeInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub rank: usize,
    pub elements: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ArraySummary>,
    /// Saga A (viz-ir-scaffold): optional IR payload describing
    /// how the inspector dialog should render this tensor.
    /// `None` falls through to the current text body; future
    /// sagas populate it and add per-kind renderers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub viz: Option<VizNode>,
}

#[derive(Serialize)]
pub struct ArraySummary {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub histogram: Vec<usize>,
}

#[derive(Serialize)]
pub struct Stage3dEvent {
    pub step_idx: usize,
    pub label: String,
    pub output: ShapeInfo,
}

// Saga C: bumped from 1000 to 4096 so rank-3 attention
// tensors stay inlined and the viz-IR detector can build the
// heatmap payload. Covers `[H, Q, K]` up to roughly H=4 with
// Q=K=32, or H=16 with Q=K=16. Larger tensors still fall
// through to the summary-only path; in the long term the
// renderer should request values on-demand from the WASM
// session rather than pre-bundling them at trace time.
const MAX_INLINE_ELEMENTS: usize = 4096;

pub fn build_shape_info(name: String, shape: Vec<usize>, values: Option<Vec<f64>>) -> ShapeInfo {
    let elements = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    let (vals, summary) = match values {
        Some(ref v) if v.len() <= MAX_INLINE_ELEMENTS => (Some(v.clone()), None),
        Some(ref v) => (None, Some(compute_summary(v))),
        None => (None, None),
    };
    ShapeInfo {
        name,
        shape: shape.clone(),
        rank: shape.len(),
        elements,
        values: vals,
        summary,
        viz: None,
    }
}

fn compute_summary(data: &[f64]) -> ArraySummary {
    let n = data.len() as f64;
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let bins = 16;
    let range = if (max - min).abs() < 1e-12 {
        1.0
    } else {
        max - min
    };
    let mut histogram = vec![0usize; bins];
    for &v in data {
        let idx = ((v - min) / range * (bins as f64 - 1.0)).round() as usize;
        histogram[idx.min(bins - 1)] += 1;
    }
    ArraySummary {
        min,
        max,
        mean,
        std,
        histogram,
    }
}

pub fn emit(event: &Stage3dEvent) {
    let Ok(js_val) = serde_wasm_bindgen::to_value(event) else {
        return;
    };
    let Some(window) = web_sys::window() else {
        return;
    };
    let Ok(func) = js_sys::Reflect::get(&window, &"__stage3d_add_step".into()) else {
        return;
    };
    if let Some(f) = func.dyn_ref::<js_sys::Function>() {
        let _ = f.call1(&window, &js_val);
    }
}

/// Parse shape from MLPL output. Handles two formats:
/// 1. Summary: `<DenseArray shape=[10, 10] elems=100 ...>`
/// 2. Values: space-separated rows (`0 1 2\n3 4 5`)
pub fn shape_from_output(output: &str) -> (Vec<usize>, usize) {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return (vec![], 0);
    }
    if let Some(parsed) = parse_dense_array_summary(trimmed) {
        return parsed;
    }
    let lines: Vec<&str> = trimmed
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let cols = lines[0].split_whitespace().count();
    let elements = lines.len() * cols;
    match (lines.len(), cols) {
        (1, 1) => (vec![], 1),
        (1, _) => (vec![cols], elements),
        _ => (vec![lines.len(), cols], elements),
    }
}

fn parse_dense_array_summary(s: &str) -> Option<(Vec<usize>, usize)> {
    if !s.starts_with("<DenseArray") {
        return None;
    }
    let shape_start = s.find("shape=[")? + 7;
    let shape_end = s[shape_start..].find(']')? + shape_start;
    let shape: Vec<usize> = s[shape_start..shape_end]
        .split(',')
        .filter_map(|p| p.trim().parse().ok())
        .collect();
    let elems = s
        .find("elems=")
        .and_then(|i| s[i + 6..].split_whitespace().next()?.parse().ok())
        .unwrap_or_else(|| shape.iter().product());
    Some((shape, elems))
}
