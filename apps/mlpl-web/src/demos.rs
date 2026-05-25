pub struct Demo {
    pub name: &'static str,
    /// One-to-three-sentence framing shown before the demo runs:
    /// what the demo does and why. Intentionally short -- the
    /// code is the real lesson.
    pub intro: &'static str,
    /// One-to-three-sentence takeaway shown after the demo's last
    /// line completes: what the output proves and where to go
    /// next. Paired with `intro` to bookend the run.
    pub takeaway: &'static str,
    pub lines: &'static [&'static str],
}

/// A heads-up rendered before a single long-running demo line.
/// Browser WASM evaluates each line on the main thread, so a
/// 30-step train block (Tiny LM) blocks the event loop for
/// seconds. Without a note the user sees a previous line's
/// output, then a stalled tab, then the result. The note paints
/// before the line starts so the wait is intentional and
/// estimated, not mysterious.
#[derive(Clone, Copy)]
pub struct ProgressNote {
    /// Demo's `name` field.
    pub demo: &'static str,
    /// Index into `Demo::lines` that the note precedes.
    pub line_idx: usize,
    /// Short heading -- e.g. "Training the language model".
    pub heading: &'static str,
    /// One-to-three-sentence body explaining what the runtime
    /// is about to do and a rough ETA on a recent laptop.
    pub body: &'static str,
}

/// Heads-up notes for demos whose individual lines block the
/// event loop long enough that the user wonders if the page is
/// frozen. Each entry attaches to a specific demo + line index.
/// Demos not listed here render with no pre-line narration --
/// the existing intro / takeaway pair is enough.
///
/// ETA wording is approximate (modern laptop, no MLX). The
/// numbers err on the high side so a faster machine never
/// sees the heads-up linger longer than the actual op.
pub const PROGRESS_NOTES: &[ProgressNote] = &[
    ProgressNote {
        demo: "Tiny LM Generate",
        line_idx: 1,
        heading: "Training the BPE tokenizer",
        body: "train_bpe walks the corpus and learns 260 byte-pair merges. A few seconds; the corpus is small but the merge loop is O(merges * pairs).",
    },
    ProgressNote {
        demo: "Tiny LM Generate",
        line_idx: 9,
        heading: "Training the language model (~10-30s)",
        body: "30 [[Adam]] steps over a 1-layer transformer (V=260, d=16, block=8). Each step runs a forward pass, a backward pass through the autograd tape, and an Adam update over every model parameter. The browser tab is single-threaded WASM so the page is unresponsive for the duration; this is normal.",
    },
    ProgressNote {
        demo: "Tiny LM Generate",
        line_idx: 13,
        heading: "Generating 20 tokens (~3-8s)",
        body: "Each token is a forward pass on the growing sequence, top_k restriction, and a multinomial sample. The output is decoded BPE bytes back to text once the loop finishes.",
    },
    ProgressNote {
        demo: "Tiny LM",
        line_idx: 1,
        heading: "Training the BPE tokenizer",
        body: "train_bpe walks the corpus and learns 260 byte-pair merges. A few seconds.",
    },
    ProgressNote {
        demo: "Tiny LM",
        line_idx: 9,
        heading: "Training the language model (~10-30s)",
        body: "30 [[Adam]] steps over a 1-layer transformer (V=260, d=16, block=8). The page is unresponsive while WASM runs on the main thread; the loss curve renders once the train block returns.",
    },
    ProgressNote {
        demo: "Tiny MLP",
        line_idx: 10,
        heading: "Training the hand-rolled MLP (~5-10s)",
        body: "600 iterations of an explicit forward + backward pass on 80 points. No autograd -- every gradient is written out so you can see the chain rule run.",
    },
    ProgressNote {
        demo: "Moons MLP",
        line_idx: 7,
        heading: "Training with Adam (~10-20s)",
        body: "200 train steps on 120 points through the autograd tape. The decision-boundary surface that follows is the visible payoff.",
    },
    ProgressNote {
        demo: "Logistic Regression",
        line_idx: 7,
        heading: "Gradient descent (~2-5s)",
        body: "300 iterations of explicit logistic-regression gradient descent on four points. Short but visibly blocks the tab.",
    },
    ProgressNote {
        demo: "Softmax Classifier",
        line_idx: 9,
        heading: "Training the softmax classifier (~3-8s)",
        body: "300 explicit-gradient steps over 90 points and 3 classes. The decision-boundary plot at the end shows the three wedges this minimization carved out.",
    },
    ProgressNote {
        demo: "Pets: cat vs dog (quick)",
        line_idx: 33,
        heading: "Training the Vision Transformer (~30-60s)",
        body: "30 full-batch [[Adam]] steps on the 8 pet images you just assembled. Each step runs the full forward pipeline (patchify -> linear-embed -> rank-3 attention -> first-token pooling -> 2-layer MLP classifier), pushes a tape of ~hundreds of nodes, runs the backward pass to compute gradients on every model parameter, then applies one Adam update. WASM is single-threaded so the page is unresponsive for the duration; this is normal, not a hang. The loss curve renders right after the train block returns -- you'll see cross-entropy drop from ~0.69 (random) toward 0 (perfect overfit on 8 images).",
    },
    ProgressNote {
        demo: "Pets: predict + gallery",
        line_idx: 61,
        heading: "Training the Vision Transformer (6 chunks x 5 steps, ~30-60s)",
        body: "30 full-batch [[Adam]] steps on 16 images, run as 6 separate train blocks of 5 steps each. Each chunk runs synchronously in WASM (~5s of unresponsive UI); the demo runner yields to the browser between chunks so the tab stays responsive end-to-end and you can see incremental progress. Adam momentum persists across chunks via the session env, so this is mathematically equivalent to one `train 30`. After training, the model predicts labels for all 16 images and renders a labeled gallery (actual / predicted under each thumbnail). 0 = cat, 1 = dog.",
    },
    ProgressNote {
        demo: "Pets: predict + gallery",
        line_idx: 73,
        heading: "Rendering the loss curve + labeled gallery",
        body: "Training is done. Below: the 30-step cross-entropy loss curve (concatenated across the six train chunks), then `predict_batch` runs the trained model over all 16 images, then `svg(X, \"gallery\", preds_2col)` renders the labeled thumbnail grid. Misclassifications stand out because the two captions under a thumbnail disagree (0/1 or 1/0).",
    },
    ProgressNote {
        demo: "Pets: multi-head ViT (quick + viz)",
        line_idx: 29,
        heading: "Training the 4-head Vision Transformer (~60-90s)",
        body: "30 full-batch [[Adam]] steps on the 8 pet images. Same forward pipeline as the single-head quick demo, but `attention(128, 4)` runs four independent attention heads in parallel and `Tensor::stack` joins their per-head [T, d/h]=[16, 32] outputs back to [16, 128]. Backward fans through every head separately. The tab is unresponsive during the train block; this is normal, not a hang.",
    },
    ProgressNote {
        demo: "Pets: multi-head ViT (quick + viz)",
        line_idx: 41,
        heading: "Rendering the four per-head attention maps",
        body: "Training is done. `attention_weights(attn, test_tokens)` returns a [4, 16, 16] tensor -- one [16, 16] softmax matrix per head over the 16 image patches. `svg(attn_maps, \"heatmap_grid\")` lays out a 2x2 grid of heatmaps with per-cell colormaps. Each cell shows what its head learned to pay attention to: row i column j is how much head h's i-th patch attends to the j-th patch.\n\nWhat to look for: heads are NOT identical. One typically concentrates on a single column (a 'this patch is the salient one' signal); another spreads attention evenly (an 'aggregate everything' signal); the remaining two pick intermediate patterns. Compare with the untrained 'ViT Multi-Head Attention Pattern' demo where all four heads look uniformly random -- the differences here are entirely the work of gradient descent.",
    },
];

/// Look up progress notes for `(demo_name, line_idx)`. Returns
/// the matching slice (usually 0 or 1 entries) without
/// allocating; callers iterate.
pub fn progress_notes_for(
    demo_name: &str,
    line_idx: usize,
) -> impl Iterator<Item = &'static ProgressNote> {
    PROGRESS_NOTES
        .iter()
        .filter(move |n| n.demo == demo_name && n.line_idx == line_idx)
}

// Demos are listed alphabetically by `name`.
pub const DEMOS: &[Demo] = &[
    crate::demos_basics::ANALYSIS_HELPERS,
    crate::demos_attention::ATTENTION_PATTERN,
    crate::demos_attention::SELF_ATTENTION_FROM_SCRATCH,
    crate::demos_attention::MULTI_HEAD_ATTENTION_FROM_SCRATCH,
    crate::demos_attention::CROSS_ATTENTION_FROM_SCRATCH,
    crate::demos_attention::ENCODER_BLOCK,
    crate::demos_attention::DECODER_BLOCK,
    crate::demos_basics::BASICS,
    crate::demos_models::DECISION_BOUNDARY_LOGICAL_GATES,
    crate::demos_models::DECISION_BOUNDARY_XOR,
    crate::demos_models::KMEANS,
    crate::demos_models::LOGISTIC_REGRESSION,
    crate::demos_basics::LOSS_CURVE,
    crate::demos_basics::MATH_FUNCTIONS,
    crate::demos_basics::MATRIX_OPS,
    crate::demos_models::MOONS_MLP,
    crate::demos_models::PCA,
    crate::demos_dim_reduction::PCA_3D,
    crate::demos_dim_reduction::PCA_LOADINGS,
    crate::demos_models::SOFTMAX_CLASSIFIER,
    crate::demos_dim_reduction::DIM_REDUCTION_ZOO,
    crate::demos_dim_reduction::UMAP_VS_PCA,
    crate::demos_dim_reduction::UMAP_VS_TSNE,
    // The interactive `Tiny LM` and `Tiny LM Generate` demos use a
    // smaller configuration than `demos/tiny_lm.mlpl` (V=280, d=32,
    // 200 steps). See demos_lm.rs.
    crate::demos_lm::TINY_LM_GENERATE,
    crate::demos_lm::TINY_LM,
    crate::demos_lm::TINY_MLP,
    crate::demos_basics::WORKSPACE_INTROSPECTION,
    crate::demos_basics::VISUALIZATIONS,
    crate::demos_vit::VIT_ATTENTION_PATTERN,
    crate::demos_vit::PETS_CAT_VS_DOG_QUICK,
    crate::demos_vit::PETS_PREDICT_GALLERY,
    crate::demos_vit::VIT_MULTI_HEAD_ATTENTION_PATTERN,
    crate::demos_vit::PETS_MULTI_HEAD_VIT,
    crate::demos_vit::PETS_ATTENTION_OVERLAY,
];

#[cfg(test)]
mod tests {
    //! Web demo smoke. Walks every entry in `DEMOS`, lexes +
    //! parses + evals each line in fresh shared environment to
    //! catch syntax / runtime drift the moment a demo string in
    //! this file falls behind language changes. Mirrors what the
    //! "Run demo" button does in the browser, minus the
    //! visualization output.
    //!
    //! Skipped: REPL slash-commands (`:tags x`) and pure
    //! comments are not parseable as expressions; the
    //! browser handles those as side-channel commands. For
    //! long-running training demos we follow the same split as
    //! `all_demos_smoke`: a quick test exercises everything
    //! except the heavy ones, and a `#[ignore]`-gated test
    //! covers the heavies on demand.
    //!
    //! `PROGRESS_NOTES` invariant: every entry's `demo` matches
    //! a real `Demo::name` and `line_idx` is within that demo's
    //! `lines` length. A mismatch would silently fail to render
    //! the heads-up note in the browser.
    use super::{DEMOS, PROGRESS_NOTES};
    use mlpl_eval::{Environment, eval_program_value};
    use mlpl_parser::{lex, parse};
    use std::collections::HashSet;

    /// Demos that call external services or do heavy training.
    const SKIP_DEMOS: &[&str] = &[
        "LLM Tool Use",
        "MLX Remote Runner",
        "Tiny LM",
        "Tiny LM Generate",
        "Moons MLP",
        "Circles MLP",
        "Transformer Block",
        "Pets: cat vs dog (quick)",
        "Pets: predict + gallery",
        "Pets: multi-head ViT (quick + viz)",
    ];

    fn run_demo(demo_name: &str, lines: &[&str]) -> Result<(), String> {
        let mut env = Environment::new();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty()
                || trimmed.starts_with("//")
                || trimmed.starts_with('#')
                || (trimmed.starts_with(':') && !trimmed.starts_with("::"))
            {
                continue;
            }
            let toks = lex(line).map_err(|e| format!("[{demo_name} line {i}] lex: {e:?}"))?;
            let prog = parse(&toks).map_err(|e| format!("[{demo_name} line {i}] parse: {e:?}"))?;
            eval_program_value(&prog, &mut env)
                .map_err(|e| format!("[{demo_name} line {i}] eval: {e:?}"))?;
        }
        Ok(())
    }

    #[test]
    fn every_quick_web_demo_runs() {
        let mut failures: Vec<String> = Vec::new();
        for demo in DEMOS.iter() {
            if SKIP_DEMOS.contains(&demo.name) {
                continue;
            }
            if let Err(msg) = run_demo(demo.name, demo.lines) {
                failures.push(msg);
            }
        }
        assert!(
            failures.is_empty(),
            "{} web demo(s) regressed:\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }

    #[test]
    #[ignore = "heavy training demos take 30+s; run with --ignored"]
    fn every_heavy_web_demo_runs() {
        let mut failures: Vec<String> = Vec::new();
        for demo in DEMOS.iter() {
            if !SKIP_DEMOS.contains(&demo.name) {
                continue;
            }
            if matches!(demo.name, "LLM Tool Use" | "MLX Remote Runner") {
                continue;
            }
            if let Err(msg) = run_demo(demo.name, demo.lines) {
                failures.push(msg);
            }
        }
        assert!(
            failures.is_empty(),
            "{} heavy web demo(s) regressed:\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }

    #[test]
    fn progress_notes_reference_real_demo_lines() {
        let demos: HashSet<&str> = DEMOS.iter().map(|d| d.name).collect();
        let mut bad: Vec<String> = Vec::new();
        for note in PROGRESS_NOTES.iter() {
            if !demos.contains(note.demo) {
                bad.push(format!("unknown demo {:?}", note.demo));
                continue;
            }
            let demo = DEMOS.iter().find(|d| d.name == note.demo).unwrap();
            if note.line_idx >= demo.lines.len() {
                bad.push(format!(
                    "{}: line_idx {} >= lines.len() {}",
                    note.demo,
                    note.line_idx,
                    demo.lines.len()
                ));
            }
        }
        assert!(
            bad.is_empty(),
            "PROGRESS_NOTES drift:\n  - {}",
            bad.join("\n  - ")
        );
    }
}
