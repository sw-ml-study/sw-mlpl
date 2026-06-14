//! Aggregated `DEMOS` list spanning the three demo sub-crates
//! plus this facade crate's own 5 modules. Saga 82 carved the
//! original `crate::demos::DEMOS` const out of mlpl-web into
//! this aggregator so each cluster (basic, vision, this
//! crate's autoencoder/gan/models/dim_reduction/rnn) lives in
//! its own sub-crate while the demo registry stays a single
//! flat slice.

use mlpl_web_demos_basic::{basics, lm, udf};
use mlpl_web_demos_types::Demo;
use mlpl_web_demos_vision::{attention, cnn, vit};

use crate::{autoencoder, dim_reduction, gan, models, rnn};

// Demos are listed alphabetically by `name`.
pub const DEMOS: &[Demo] = &[
    lm::ASK_OLLAMA_CONTEXTUAL,
    lm::CUDA_LORA_FINETUNE,
    lm::CUDA_TICTACTOE_FINETUNE,
    lm::DIFFUSION_2D,
    lm::MLX_LORA_FINETUNE,
    lm::MLX_TICTACTOE_FINETUNE,
    basics::ANALYSIS_HELPERS,
    attention::ATTENTION_PATTERN,
    attention::SELF_ATTENTION_FROM_SCRATCH,
    attention::MULTI_HEAD_ATTENTION_FROM_SCRATCH,
    attention::CROSS_ATTENTION_FROM_SCRATCH,
    attention::ENCODER_BLOCK,
    attention::DECODER_BLOCK,
    autoencoder::AUTOENCODER,
    basics::BASICS,
    cnn::SIMPLE_CNN,
    models::DECISION_BOUNDARY_LOGICAL_GATES,
    models::DECISION_BOUNDARY_XOR,
    models::HOW_GRADIENT_DESCENT_WORKS,
    models::KMEANS,
    models::LOGISTIC_REGRESSION,
    basics::LOSS_CURVE,
    basics::MATH_FUNCTIONS,
    basics::MATRIX_OPS,
    models::MOONS_MLP,
    models::PCA,
    dim_reduction::PCA_3D,
    dim_reduction::PCA_LOADINGS,
    rnn::RNN_SEQUENCE,
    rnn::LSTM_SEQUENCE,
    gan::GAN_2D,
    models::SOFTMAX_CLASSIFIER,
    dim_reduction::DIM_REDUCTION_ZOO,
    dim_reduction::UMAP_VS_PCA,
    dim_reduction::UMAP_VS_TSNE,
    // The interactive `Tiny LM` and `Tiny LM Generate` demos use a
    // smaller configuration than `demos/tiny_lm.mlpl` (V=280, d=32,
    // 200 steps). See lm.rs.
    lm::TINY_LM_GENERATE,
    lm::TINY_LM,
    lm::TINY_MLP,
    basics::WORKSPACE_INTROSPECTION,
    basics::UPLOAD_INSPECT,
    udf::USER_FUNCTIONS,
    basics::VISUALIZATIONS,
    models::WATCH_MODEL_LEARN,
    basics::BPE_ATTENTION_LABELS,
    vit::VIT_ATTENTION_PATTERN,
    vit::PETS_CAT_VS_DOG_QUICK,
    vit::PETS_PREDICT_GALLERY,
    vit::VIT_MULTI_HEAD_ATTENTION_PATTERN,
    vit::PETS_MULTI_HEAD_VIT,
    vit::PETS_ATTENTION_OVERLAY,
];
