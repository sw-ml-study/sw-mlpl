//! Learning paths: curated ordered walks through the
//! tutorial / demo / diagram / glossary surfaces.
//!
//! A `LearningPath` is just a list of `Step`s, each of which
//! references existing content by name (lessons by title,
//! demos by name, diagrams by filename slug, glossary entries
//! by exact term). The walker view (`paths_view::PathsView`)
//! renders each step with a path-specific "why this is here"
//! framing and -- for lessons / demos -- a button that jumps
//! to the corresponding tab. Diagrams and glossary entries
//! render inline.
//!
//! Paths are pure data: adding a new path is one entry in
//! `PATHS` below, no UI changes needed.

#[derive(Clone, Copy, PartialEq)]
pub struct LearningPath {
    pub title: &'static str,
    pub blurb: &'static str,
    pub steps: &'static [Step],
}

#[derive(Clone, Copy, PartialEq)]
pub enum Step {
    /// A tutorial lesson, looked up by exact title.
    Lesson {
        title: &'static str,
        why: &'static str,
    },
    /// A demo, looked up by exact name.
    Demo {
        name: &'static str,
        why: &'static str,
    },
    /// A diagram, looked up by filename slug (matching the
    /// numbered `<slug>.svg` files in `diagrams/`).
    Diagram {
        slug: &'static str,
        why: &'static str,
    },
    /// A glossary entry, looked up by exact term (matching
    /// `## TermName` headers in `docs/glossary.md`).
    Glossary {
        term: &'static str,
        why: &'static str,
    },
    /// A path-orientation note that does not reference
    /// existing content. Shown as a small framing card.
    Note {
        title: &'static str,
        body: &'static str,
    },
}

pub const PATHS: &[LearningPath] = &[
    LearningPath {
        title: "Zero to LLM",
        blurb: "The spine: orientation -> arrays -> classifiers -> MLP -> autograd -> attention -> transformer -> tiny LM. Twelve steps; assume zero ML background.",
        steps: &[
            Step::Lesson {
                title: "What is ML, and why are we here?",
                why: "Set the destination first. Every later step is a variation of the same recipe (data, model, loss, gradient descent).",
            },
            Step::Lesson {
                title: "Hello Numbers",
                why: "MLPL's smallest possible expressions. Numbers and operators -- the substrate every other lesson is built on.",
            },
            Step::Lesson {
                title: "Arrays",
                why: "Vectors, then matrices. APL-derived shape semantics that ML inherits.",
            },
            Step::Lesson {
                title: "Matrices",
                why: "Reshape, transpose, dimension manipulation. The shape-arithmetic layer ML stands on.",
            },
            Step::Lesson {
                title: "Math and Activations",
                why: "exp, log, sigmoid, tanh -- the elementwise primitives every neural layer composes.",
            },
            Step::Lesson {
                title: "Machine Learning: Logistic Regression",
                why: "The hello-world ML model: fit two weights to four points using hand-rolled gradient descent. Forward + backward pass written out explicitly.",
            },
            Step::Lesson {
                title: "Going Non-Linear: A Tiny MLP",
                why: "Add a hidden layer + tanh. Solves problems no linear model can. The chain-rule backward pass is visible in the code.",
            },
            Step::Lesson {
                title: "Automatic Differentiation",
                why: "Replace the hand-rolled chain rule with `grad(loss, wrt)`. The lift from manual derivation to automatic differentiation that backprop unlocked in 1986.",
            },
            Step::Diagram {
                slug: "12_attention",
                why: "Visual reference for scaled-dot-product attention before reading the from-scratch implementation. The whole formula in one diagram.",
            },
            Step::Lesson {
                title: "Self-Attention from Scratch",
                why: "Build one head of attention from primitives -- three projections, score, softmax, weighted sum. The transformer's core in 15 lines.",
            },
            Step::Diagram {
                slug: "17_gpt_decoder_only",
                why: "Where a single attention layer fits in a stacked decoder-only transformer. Visualizes what \"Tiny LM\" actually instantiates.",
            },
            Step::Demo {
                name: "Tiny LM Generate",
                why: "End-to-end: BPE tokenizer + 1-layer transformer LM trained 30 steps on a tiny corpus, then sampled to generate text. The smallest program that learns to talk.",
            },
        ],
    },
    LearningPath {
        title: "Visual: ML by diagram",
        blurb: "Walk all 38 ML reference diagrams in numbered order. Pure browse path -- no MLPL code -- gives you the whole concept map before you dig into any single piece.",
        steps: &[
            Step::Note {
                title: "How to use this path",
                body: "Each step is one diagram. The blurb says what slice of MLPL covers the same ground (or notes that we have only the glossary entry, not a runnable demo). Skim the whole path first to get the lay of the land; come back to specific diagrams when you start a topic.",
            },
            Step::Diagram {
                slug: "01_linear_regression",
                why: "y = wX + b + MSE + gradient descent. The smallest-possible ML loop.",
            },
            Step::Diagram {
                slug: "02_logistic_regression",
                why: "Add a sigmoid + cross-entropy. Now it is a classifier. We have the demo + lesson.",
            },
            Step::Diagram {
                slug: "03_decision_tree",
                why: "Greedy yes/no splits on features. Glossary entry only -- no MLPL primitive.",
            },
            Step::Diagram {
                slug: "04_random_forest",
                why: "Bagged ensemble of decision trees. Glossary only.",
            },
            Step::Diagram {
                slug: "05_svm",
                why: "Maximum-margin hyperplane + kernel trick. Pre-deep-learning state of the art. Glossary only.",
            },
            Step::Diagram {
                slug: "06_perceptron",
                why: "Rosenblatt 1958 -- one linear layer + threshold. We use it in the History of ML lesson.",
            },
            Step::Diagram {
                slug: "07_mlp",
                why: "Stack linear + nonlinearity. Tiny MLP demo + lesson cover this.",
            },
            Step::Diagram {
                slug: "08_cnn",
                why: "Conv + pool + FC. Glossary only -- no conv2d primitive in MLPL.",
            },
            Step::Diagram {
                slug: "09_resnet",
                why: "y = x + f(x). MLPL has `residual(...)` directly; the encoder/decoder block lessons use it.",
            },
            Step::Diagram {
                slug: "10_rnn",
                why: "Hidden state passed through time. Glossary only -- deferred.",
            },
            Step::Diagram {
                slug: "11_lstm",
                why: "Gated RNN cell. Glossary only -- deferred.",
            },
            Step::Diagram {
                slug: "12_attention",
                why: "softmax(Q K^T / sqrt(d_k)) V. Self-Attention from Scratch lesson + Attention Pattern demo.",
            },
            Step::Diagram {
                slug: "13_multi_head_attention",
                why: "h heads on d_k slabs. Multi-Head Attention from Scratch lesson + demo.",
            },
            Step::Diagram {
                slug: "14_transformer_encoder",
                why: "Stack of encoder blocks. Encoder Block lesson + demo.",
            },
            Step::Diagram {
                slug: "15_transformer_decoder",
                why: "Causal self-attn + cross-attn + FFN. Decoder Block lesson + demo.",
            },
            Step::Diagram {
                slug: "16_encoder_decoder_transformer",
                why: "Full seq-to-seq. We have the parts (encoder, decoder); no end-to-end demo yet.",
            },
            Step::Diagram {
                slug: "17_gpt_decoder_only",
                why: "Tiny LM IS this: stacked causal-self-attn blocks.",
            },
            Step::Diagram {
                slug: "18_moe",
                why: "k-of-N routed experts per FFN. Glossary only.",
            },
            Step::Diagram {
                slug: "19_rag",
                why: "Retrieve docs, prepend, generate. Glossary only -- deferred.",
            },
            Step::Diagram {
                slug: "20_agent_loop",
                why: "LLM tool-use cycle. Glossary entry; `llm_call` is the building block.",
            },
            Step::Diagram {
                slug: "21_vit",
                why: "Patches as tokens. Glossary only -- needs image inputs.",
            },
            Step::Diagram {
                slug: "22_unet",
                why: "Conv encoder-decoder + skips. Glossary only.",
            },
            Step::Diagram {
                slug: "23_diffusion",
                why: "Iterative denoising. Glossary only -- deferred.",
            },
            Step::Diagram {
                slug: "24_clip",
                why: "Dual-encoder image + text. Glossary only -- needs image inputs.",
            },
            Step::Diagram {
                slug: "25_vlm",
                why: "Vision encoder + projector + LM. Glossary only.",
            },
            Step::Diagram {
                slug: "26_mamba_ssm",
                why: "Selective state-space alternative to attention. Glossary only -- deferred.",
            },
            Step::Diagram {
                slug: "27_training_loop",
                why: "Forward -> loss -> backward -> step. Every training demo IS this.",
            },
            Step::Diagram {
                slug: "28_backprop",
                why: "Reverse-mode chain rule. Why backprop? lesson + Automatic Differentiation lesson cover this.",
            },
            Step::Diagram {
                slug: "29_data_parallel_training",
                why: "Replicate model, split batch, all-reduce. Glossary only.",
            },
            Step::Diagram {
                slug: "30_tensor_parallel_training",
                why: "Split layer weights across devices. Glossary only.",
            },
            Step::Diagram {
                slug: "31_pipeline_parallel_training",
                why: "Split layers across devices. Glossary only.",
            },
            Step::Diagram {
                slug: "32_lora",
                why: "Low-rank adapters on a frozen base. LoRA Fine-Tuning lesson covers this.",
            },
            Step::Diagram {
                slug: "33_qlora",
                why: "Int4 base + bf16 LoRA. Glossary only -- LoRA exists, quantization does not.",
            },
            Step::Diagram {
                slug: "34_rlhf",
                why: "SFT -> reward model -> PPO. Glossary only -- deferred.",
            },
            Step::Diagram {
                slug: "35_dpo",
                why: "Direct preference optimization. Glossary only.",
            },
            Step::Diagram {
                slug: "36_self_play_training",
                why: "Agent generates its own training signal. Glossary only -- deferred.",
            },
            Step::Diagram {
                slug: "37_grokking",
                why: "Delayed generalization after long memorization. Glossary only -- a research curiosity.",
            },
            Step::Diagram {
                slug: "38_superposition",
                why: "Networks pack more features than dimensions. Glossary only -- mechanistic interpretability.",
            },
        ],
    },
    LearningPath {
        title: "Build a transformer from primitives",
        blurb: "The from-scratch attention bundle: see the diagram, then build the same thing in MLPL with no model-DSL shortcuts. Nine steps; alternates diagrams and lessons.",
        steps: &[
            Step::Diagram {
                slug: "12_attention",
                why: "Single-head self-attention as one diagram. Frame the math before the code.",
            },
            Step::Lesson {
                title: "Self-Attention from Scratch",
                why: "Build one head: three projections (Q/K/V) + scaled dot-product + softmax + weighted sum. Every line has a paragraph hover tooltip.",
            },
            Step::Diagram {
                slug: "13_multi_head_attention",
                why: "How `h` heads run in parallel on `d_k = d_model/h` slabs.",
            },
            Step::Lesson {
                title: "Multi-Head Attention from Scratch",
                why: "Build the multi-head version using selector matrices for column slicing -- MLPL has no surface slice op, so the slabbing is explicit.",
            },
            Step::Lesson {
                title: "Cross-Attention from Scratch",
                why: "Same formula, but Q comes from a target sequence and K/V come from a separate source. Non-square weight heatmap is the visual signature.",
            },
            Step::Diagram {
                slug: "14_transformer_encoder",
                why: "How attention layers compose with FFN + residuals into a stackable encoder block.",
            },
            Step::Lesson {
                title: "Encoder Block",
                why: "Build one encoder block via the model DSL: chain(residual(rms_norm + self-attn), residual(rms_norm + ffn)).",
            },
            Step::Diagram {
                slug: "15_transformer_decoder",
                why: "Decoder = encoder + cross-attention sub-block. The third sub-block is the only difference.",
            },
            Step::Lesson {
                title: "Decoder Block",
                why: "Build the full three-sub-block decoder: causal self-attn + cross-attn (from scratch) + FFN. After this you have all the pieces of a real transformer.",
            },
        ],
    },
];
