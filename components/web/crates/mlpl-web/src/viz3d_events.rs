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

const MAX_INLINE_ELEMENTS: usize = 1000;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_serializes() {
        let ev = Stage3dEvent {
            step_idx: 0,
            label: "x = 1 + 2".into(),
            output: build_shape_info("x".into(), vec![], None),
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"step_idx\":0"));
        assert!(json.contains("\"rank\":0"));
    }

    #[test]
    fn shape_scalar() {
        assert_eq!(shape_from_output("3"), (vec![], 1));
    }

    #[test]
    fn shape_vector() {
        assert_eq!(shape_from_output("0 1 2 3 4"), (vec![5], 5));
    }

    #[test]
    fn shape_matrix() {
        let out = "0 1 2 3\n4 5 6 7\n8 9 10 11";
        assert_eq!(shape_from_output(out), (vec![3, 4], 12));
    }

    #[test]
    fn shape_dense_array_summary() {
        let out = "<DenseArray shape=[10, 10] elems=100 first=[0, 1, 2, ...]>";
        assert_eq!(shape_from_output(out), (vec![10, 10], 100));
    }

    #[test]
    fn shape_dense_array_vector() {
        let out = "<DenseArray shape=[100] elems=100 first=[0, 1, 2, ...]>";
        assert_eq!(shape_from_output(out), (vec![100], 100));
    }
}
