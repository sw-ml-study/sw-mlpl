//! Hardcoded structural specs for the supported hypothetical
//! models. Each entry's dims come from the model's published
//! config; parameter-count totals line up within ~30% of the
//! reported model size.

use crate::error::FeasibilityError;

pub(crate) struct HypSpec {
    pub vocab: f64,
    pub d_model: f64,
    pub layers: f64,
    pub intermediate: f64,
}

pub(crate) fn lookup_hypothetical(name: &str) -> Result<HypSpec, FeasibilityError> {
    let spec = match name {
        "smollm-135m" => HypSpec {
            vocab: 49152.0,
            d_model: 576.0,
            layers: 30.0,
            intermediate: 1536.0,
        },
        "smollm-360m" => HypSpec {
            vocab: 49152.0,
            d_model: 960.0,
            layers: 32.0,
            intermediate: 2560.0,
        },
        "smollm-1.7b" => HypSpec {
            vocab: 49152.0,
            d_model: 2048.0,
            layers: 24.0,
            intermediate: 8192.0,
        },
        "llama-3.2-1b" => HypSpec {
            vocab: 128256.0,
            d_model: 2048.0,
            layers: 16.0,
            intermediate: 8192.0,
        },
        "qwen-2.5-0.5b" => HypSpec {
            vocab: 151936.0,
            d_model: 896.0,
            layers: 24.0,
            intermediate: 4864.0,
        },
        other => return Err(FeasibilityError::UnknownModel(other.into())),
    };
    Ok(spec)
}
