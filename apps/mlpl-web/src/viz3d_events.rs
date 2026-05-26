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

pub fn shape_from_output(output: &str) -> (Vec<usize>, usize) {
    let trimmed = output.trim();
    if trimmed.is_empty() || !trimmed.starts_with('[') {
        return (vec![], 0);
    }
    let rank = trimmed.bytes().take_while(|&b| b == b'[').count();
    let flat: Vec<&str> = trimmed
        .replace(['[', ']'], " ")
        .split_whitespace()
        .filter(|s| !s.is_empty() && !s.ends_with(','))
        .chain(
            trimmed
                .replace(['[', ']'], " ")
                .split_whitespace()
                .filter(|s| s.ends_with(',')),
        )
        .collect();
    let elements = trimmed
        .matches(|c: char| c.is_ascii_digit() || c == '.' || c == '-')
        .count()
        .max(1);
    let shape = match rank {
        0 => vec![],
        1 => vec![count_top_elements(trimmed)],
        _ => infer_shape(trimmed, rank),
    };
    (shape, elements)
}

fn count_top_elements(s: &str) -> usize {
    s.trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter(|p| !p.trim().is_empty())
        .count()
}

fn infer_shape(s: &str, rank: usize) -> Vec<usize> {
    if rank == 2 {
        let rows = s.trim_start_matches('[').trim_end_matches(']');
        let row_count = rows.matches('[').count();
        let first_row = rows.split(']').next().unwrap_or("");
        let cols = count_top_elements(first_row);
        vec![row_count, cols]
    } else {
        vec![rank]
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
}
