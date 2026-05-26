use serde::Serialize;
use wasm_bindgen::JsCast;

#[derive(Serialize)]
pub struct ShapeInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub rank: usize,
    pub elements: usize,
}

#[derive(Serialize)]
pub struct Stage3dEvent {
    pub step_idx: usize,
    pub label: String,
    pub output: ShapeInfo,
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

/// Parse shape from MLPL's space-separated display format.
/// Scalar: "3" -> ([], 1). Vector: "0 1 2" -> ([3], 3).
/// Matrix: "0 1\n2 3" -> ([2, 2], 4).
pub fn shape_from_output(output: &str) -> (Vec<usize>, usize) {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return (vec![], 0);
    }
    let lines: Vec<&str> = trimmed
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let cols = lines[0].split_whitespace().count();
    let elements = lines.len() * cols;
    if lines.len() == 1 && cols == 1 {
        (vec![], 1)
    } else if lines.len() == 1 {
        (vec![cols], elements)
    } else {
        (vec![lines.len(), cols], elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_serializes() {
        let ev = Stage3dEvent {
            step_idx: 0,
            label: "x = 1 + 2".into(),
            output: ShapeInfo {
                name: "x".into(),
                shape: vec![],
                rank: 0,
                elements: 1,
            },
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
}
