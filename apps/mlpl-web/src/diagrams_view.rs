//! Diagrams tab in the Help dialog.
//!
//! Renders an index of every mermaid diagram shipped under
//! `diagrams/` (copied into `pages/diagrams/` at build time
//! via `<link data-trunk rel="copy-dir">`). Each entry is a
//! click-to-toggle row: clicking the title shows / hides the
//! inline `<img>` tag pointing at the SVG. Lazy-loaded via
//! `loading="lazy"` so the 8MB total set does not all fetch
//! at once.
//!
//! The mermaid SVGs use a hardcoded `background-color: white`
//! that clashes with the Catppuccin dark theme; the simplest
//! adaptation is to wrap each in a white "card" so it reads
//! as an intentional design element rather than a glitch.

use yew::prelude::*;

/// `(filename_stem, friendly_title, one_line_blurb)`. The
/// numeric prefix on the filename gives a logical ordering
/// (classical -> NN -> attention -> transformer -> LLM
/// extensions -> training infra -> fine-tuning / alignment
/// -> phenomena) so the index reads as a guided tour.
pub const DIAGRAMS: &[(&str, &str, &str)] = &[
    (
        "01_linear_regression",
        "Linear Regression",
        "y = wX + b, MSE loss, gradient descent.",
    ),
    (
        "02_logistic_regression",
        "Logistic Regression",
        "Linear layer + sigmoid + cross-entropy.",
    ),
    (
        "03_decision_tree",
        "Decision Tree",
        "Greedy yes/no splits on input features.",
    ),
    (
        "04_random_forest",
        "Random Forest",
        "Bagged ensemble of trees, voted prediction.",
    ),
    (
        "05_svm",
        "Support Vector Machine",
        "Maximum-margin hyperplane + kernel trick.",
    ),
    (
        "06_perceptron",
        "Perceptron",
        "Single linear layer + threshold (Rosenblatt 1958).",
    ),
    (
        "07_mlp",
        "Multilayer Perceptron",
        "Stacked linear + nonlinearity layers.",
    ),
    (
        "08_cnn",
        "Convolutional Neural Network",
        "Conv + pool + fully-connected stack.",
    ),
    (
        "09_resnet",
        "ResNet (Residual Network)",
        "y = x + f(x) skip connections.",
    ),
    (
        "10_rnn",
        "Recurrent Neural Network",
        "Hidden state passed through time steps.",
    ),
    (
        "11_lstm",
        "LSTM Cell",
        "RNN with input / forget / output gates.",
    ),
    (
        "12_attention",
        "Scaled Dot-Product Attention",
        "softmax(Q K^T / sqrt(d_k)) V.",
    ),
    (
        "13_multi_head_attention",
        "Multi-Head Attention",
        "h heads on d_k = d_model/h slabs in parallel.",
    ),
    (
        "14_transformer_encoder",
        "Transformer Encoder",
        "N x (self-attn + FFN) with residuals.",
    ),
    (
        "15_transformer_decoder",
        "Transformer Decoder Block",
        "Causal self-attn + cross-attn + FFN.",
    ),
    (
        "16_encoder_decoder_transformer",
        "Encoder-Decoder Transformer",
        "Full seq-to-seq (e.g. T5-style).",
    ),
    (
        "17_gpt_decoder_only",
        "GPT (Decoder-Only)",
        "Stacked causal-self-attn blocks; no cross-attn.",
    ),
    (
        "18_moe",
        "Mixture of Experts",
        "Routed sparse FFN: k of N experts active.",
    ),
    (
        "19_rag",
        "Retrieval-Augmented Generation",
        "Retriever + LLM: fetch context, prepend, generate.",
    ),
    (
        "20_agent_loop",
        "Agentic LLM Loop",
        "Plan -> tool call -> result -> repeat.",
    ),
    (
        "21_vit",
        "Vision Transformer",
        "Patches as tokens -> transformer.",
    ),
    (
        "22_unet",
        "U-Net",
        "Conv encoder-decoder with skip connections.",
    ),
    (
        "23_diffusion",
        "Diffusion Model",
        "Iterative denoising; train + sample loops.",
    ),
    (
        "24_clip",
        "CLIP",
        "Dual encoder (image + text) into shared space.",
    ),
    (
        "25_vlm",
        "Vision-Language Model",
        "Vision encoder + projector + LM.",
    ),
    (
        "26_mamba_ssm",
        "Mamba / State Space Model",
        "Selective SSM as transformer alternative.",
    ),
    (
        "27_training_loop",
        "Training Loop",
        "Forward -> loss -> backward -> step.",
    ),
    (
        "28_backprop",
        "Backpropagation",
        "Reverse-mode chain rule on the comp graph.",
    ),
    (
        "29_data_parallel_training",
        "Data Parallel Training",
        "Replicate model, split batch, all-reduce grads.",
    ),
    (
        "30_tensor_parallel_training",
        "Tensor Parallel Training",
        "Split a single layer's weights across devices.",
    ),
    (
        "31_pipeline_parallel_training",
        "Pipeline Parallel Training",
        "Split layers across devices, pipeline microbatches.",
    ),
    (
        "32_lora",
        "LoRA Fine-Tuning",
        "Low-rank adapters added to a frozen base.",
    ),
    ("33_qlora", "QLoRA", "Int4 base + bf16 LoRA adapters."),
    ("34_rlhf", "RLHF Pipeline", "SFT -> reward model -> PPO."),
    (
        "35_dpo",
        "DPO",
        "Direct preference optimization (no reward model).",
    ),
    (
        "36_self_play_training",
        "Self-Play Training",
        "Agent generates its own training signal.",
    ),
    (
        "37_grokking",
        "Grokking",
        "Delayed generalization after long memorization.",
    ),
    (
        "38_superposition",
        "Feature Superposition",
        "Network packs more features than dimensions.",
    ),
    (
        "39_patchify",
        "Patchify (ViT)",
        "64x64 image -> 16 patches -> [16, 768] tokens.",
    ),
    (
        "40_multi_head_attention",
        "Multi-Head Attention (4-head walkthrough)",
        "Q/K/V projection -> per-head SDPA -> stack -> Wo.",
    ),
    (
        "41_stack_tape_op",
        "Stack vs chained concat",
        "One stack node beats N-1 binary concat nodes.",
    ),
    (
        "42_vit_pipeline",
        "ViT forward pipeline",
        "Image -> patches -> embed + CLS + pos -> attn -> CLS -> MLP.",
    ),
    (
        "43_result_type",
        "Value::Result (Ok / Err)",
        "ok bool + payload; is_ok / unwrap / err_message / unwrap_or.",
    ),
    (
        "44_heatmap_grid",
        "heatmap_grid viz",
        "[N, R, C] tensor unfolds into a 2x2 grid of heatmaps.",
    ),
    (
        "45_upload_result_flow",
        ":upload x flow",
        "File picker -> Canvas decode -> Ok(Record) / Err(message).",
    ),
];

#[function_component(DiagramsView)]
pub fn diagrams_view() -> Html {
    let open_idx = use_state(|| Option::<usize>::None);
    html! {
        <div class="diagrams-index">
            <p class="diagrams-intro">
                { format!(
                    "{} diagrams covering ML concepts from classical regression through modern transformer training. Click any title to expand. The diagram set is an external visual reference; the matching MLPL demos / lessons / glossary entries are linked separately under the other tabs.",
                    DIAGRAMS.len(),
                ) }
            </p>
            <ol class="diagrams-list">
                { for DIAGRAMS.iter().enumerate().map(|(i, (slug, title, blurb))| {
                    let is_open = *open_idx == Some(i);
                    let on_toggle = {
                        let h = open_idx.clone();
                        Callback::from(move |_| h.set(if is_open { None } else { Some(i) }))
                    };
                    let body = if is_open {
                        let src = format!("diagrams/{slug}.svg");
                        html! {
                            <div class="diagram-card">
                                <img class="diagram-svg" src={src} alt={(*title).to_string()} loading="lazy"/>
                            </div>
                        }
                    } else {
                        html! {}
                    };
                    html! {
                        <li class="diagram-row">
                            <button class={if is_open { "diagram-toggle open" } else { "diagram-toggle" }} onclick={on_toggle}>
                                <span class="diagram-title">{ *title }</span>
                                <span class="diagram-blurb">{ *blurb }</span>
                            </button>
                            { body }
                        </li>
                    }
                }) }
            </ol>
        </div>
    }
}
