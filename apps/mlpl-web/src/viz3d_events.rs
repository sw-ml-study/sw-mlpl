use serde::Serialize;

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
    let Ok(json) = serde_json::to_string(event) else {
        return;
    };
    let escaped = json.replace('\\', "\\\\").replace('\'', "\\'");
    let _ = js_sys::eval(&format!("window.__stage3d_add_step('{escaped}')"));
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
}
