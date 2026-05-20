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
    LearningPath {
        title: "Vision Transformers in MLPL",
        blurb: "From a synthetic image to a trained multi-head cat-vs-dog classifier with per-head attention visualization, ending with uploading a photo from your phone for the trained model to classify. The same attention machinery from the transformer demos, applied to image patches. Alternates diagrams, glossary, and demos; assumes you have already seen scaled-dot-product attention.",
        steps: &[
            Step::Note {
                title: "How this path is laid out",
                body: "ViT is not new architecture -- it is the SAME [[Attention]] block from the transformer demos, fed image patches instead of token embeddings. The early steps explain the new pieces ([[patchify (builtin)]], [[take (builtin)]], batched and [[Multi-head attention]]); the demos wire them together at three depths: an untrained pattern, a trained single-head classifier with a labeled gallery, and a trained multi-head classifier with per-head attention maps. The path ends with bring-your-own-image (the [[:upload (REPL command)]] command) so you can classify a photo from your phone end-to-end. Click any underlined term to pop up its glossary entry.",
            },
            Step::Glossary {
                term: "patchify (builtin)",
                why: "ViT's one new primitive on top of standard transformer parts. Cuts an image into a sequence of flattened patches so attention can treat them as tokens.",
            },
            Step::Diagram {
                slug: "39_patchify",
                why: "How a 64x64 image becomes a [16, 768] token sequence. The first picture that ever made `patchify` click is usually this one.",
            },
            Step::Glossary {
                term: "take (builtin)",
                why: "Per-batch / per-row indexing. Used by the trained demo to extract single images from `pets_tiny.X` and to pull the first-token output of attention as a CLS-like aggregator.",
            },
            Step::Glossary {
                term: "Stack (tape op)",
                why: "Single-node, N-way concatenation along an existing axis. The multi-head autograd tape uses it to join per-head outputs (and the rank-3 path uses it to join per-batch outputs) in O(N) instead of an O(N^2) binary-concat chain.",
            },
            Step::Diagram {
                slug: "41_stack_tape_op",
                why: "Why we built Stack as a primitive instead of folding N concats. The before/after picture is the autograd cost story in one image.",
            },
            Step::Diagram {
                slug: "40_multi_head_attention",
                why: "How 4 heads share `d_model` and recombine. Pair this with the multi-head pattern demo below to read the [4, 17, 17] shape correctly.",
            },
            Step::Diagram {
                slug: "42_vit_pipeline",
                why: "The full forward path with shapes -- image -> patches -> embed + CLS + pos -> attention -> CLS -> MLP -> argmax. The map for everything that follows.",
            },
            Step::Demo {
                name: "ViT Attention Pattern (no training)",
                why: "Mechanical end-to-end forward pipeline on a synthetic image: patchify -> linear embed -> concat CLS -> + positional -> attention -> softmax -> heatmap. No training, so the heatmap is random; the demo's point is showing every Phase-1 builtin composing into the recipe.",
            },
            Step::Diagram {
                slug: "44_heatmap_grid",
                why: "How `svg(_, \"heatmap_grid\")` unfolds a [N, R, C] tensor into a grid. Required reading before the multi-head pattern demo, otherwise the four-cell layout is mystery geometry.",
            },
            Step::Demo {
                name: "ViT Multi-Head Attention Pattern (no training)",
                why: "Same pipeline but `attention(128, 4, ...)` -- four heads. attention_weights returns [4, 17, 17] and `svg(_, \"heatmap_grid\")` lays out a 2x2 grid of per-head heatmaps. Untrained, so all four look uniformly random and similar to each other; this is the baseline that the trained multi-head demo is judged against.",
            },
            Step::Glossary {
                term: "Oxford-IIIT Pet dataset",
                why: "The labeled image source. 7,393 cat/dog photos at ~500x500. MLPL ships a 200-image 64x64 subset (pets_tiny) embedded in the WASM binary so the trained demo runs without a server.",
            },
            Step::Demo {
                name: "Pets: cat vs dog (quick)",
                why: "Trained single-head Vision Transformer end-to-end: 8 balanced images, 30 adam steps, loss curve falls toward 0, training accuracy 1.0. The tail of the demo shows how to inspect the pets tensor with chained `take` calls.",
            },
            Step::Demo {
                name: "Pets: predict + gallery",
                why: "The same trained ViT, but now scaled to 16 images and run end-to-end with `predict_batch` + the 3-arg `svg(X, \"gallery\", overlay)` viz. Each thumbnail gets an `actual / predicted` caption -- you can finally look at a specific cat photo and read what the model said about it.",
            },
            Step::Demo {
                name: "Pets: multi-head ViT (quick + viz)",
                why: "The headline demo for the multi-head story. Same architecture as the single-head quick demo but `attention(128, 4)`; after 30 adam steps it renders the four post-training attention maps. Compare with the untrained multi-head pattern demo above to see what specialization gradient descent buys -- the heads start identical and end different.",
            },
            Step::Glossary {
                term: ":upload (REPL command)",
                why: "Pick a photo from your device. The browser decodes + resizes it to 64x64 and binds the result under your chosen name as `Ok({pixels, h, w})`. A cancelled or unreadable upload binds `Err(\"...\")` instead so the program can branch on success without crashing on an undefined variable.",
            },
            Step::Diagram {
                slug: "43_result_type",
                why: "`Value::Result` and its accessors -- `is_ok`, `unwrap`, `err_message`, `unwrap_or`. The shape every `:upload` reply lands in.",
            },
            Step::Diagram {
                slug: "45_upload_result_flow",
                why: "End-to-end flow: REPL :upload -> file picker -> Canvas decode -> Ok(Record) / Err(message). Read this before typing the five lines below.",
            },
            Step::Note {
                title: "Bring-your-own-image (try it now)",
                body: "After running any of the three trained pets demos above, the model bindings (`linear_p`, `attn`, `classifier`) stay in your REPL session. Then type these five lines (click any underlined term for its glossary entry):\n\n1. `:upload x` -- the [[:upload (REPL command)]] opens the file picker; pick any photo.\n2. `is_ok(x)` -- returns 1 if the upload succeeded, 0 if you dismissed the dialog or the file was not a valid image. (See the [[Result type]] for what `Ok` / `Err` mean.)\n3. `svg(unwrap(x).pixels, \"gallery\")` -- see the 64x64 resized version.\n4. `img = unwrap(x).pixels` -- pull the tensor out of the Result.\n5. `predict_batch(classifier, take(apply(attn, reshape(apply(linear_p, reshape(patchify(img, 16), [16, 768])), [1, 16, 128])), 1, 0))` -- returns `[0]` for cat or `[1]` for dog. Uses [[predict_batch (builtin)]] over a [[patchify (builtin)]]-and-attention forward pass.\n\nIf you dismissed the picker, `x = Err(\"cancelled\")` and `is_ok(x)` returns 0. Other Err flavors: `\"decode failed: not a valid image\"` (binary renamed as .jpg), `\"read failed\"` (zero-byte / permissions). Read the diagnostic with `err_message(x)`.\n\nReality check: the in-browser models train on only 8-20 images, so they memorize the training set and tend to classify any unseen photo as a cat. See `docs/better-cat-dog-future-demos.md` for the recommended improvement ladder (full-pets_tiny + val split, confidence threshold, augmentation, etc.).",
            },
            Step::Note {
                title: "Beyond this path",
                body: "**Full-resolution demo.** `demos/vit_multihead_thorough.mlpl` trains the same architecture on 128x128 input from the full Oxford-IIIT Pet (via `fetch_dataset`) inside a `device(\"mlx\")` block. Runs on Apple Silicon with `--features mlx`; falls back to CPU on any other host.\n\n**Pending architectural pieces.** Layer norm with learned affine and the tanh-approximation GELU are not yet builtins. Adding them would close the architectural gap to the upstream ViT notebook.\n\n**Making the classifier actually work on unseen photos.** Today's in-browser trained models overfit hard on 8-20 images and tend to classify every uploaded photo as a cat. The 7-demo improvement ladder in `docs/better-cat-dog-future-demos.md` lays out the recommended sequence: full-pets_tiny + held-out validation split first (no new builtins), then confidence-thresholded \"other\" output, then horizontal-flip augmentation, then AdamW, then LayerNorm + GELU, then the thorough MLX run.",
            },
        ],
    },
];
