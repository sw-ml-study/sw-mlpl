//! Shared demo types for the web playground demo cluster: the
//! `Demo` struct, `ProgressNote` with `PROGRESS_NOTES`, and
//! `progress_notes_for`. Extracted into mlpl-web-demos-types
//! (saga 82) so every themed sub-crate (basic, vision, the
//! facade's own demos) references one shared `Demo` struct
//! without depending on a sibling demo crate.

pub struct Demo {
    pub name: &'static str,
    pub category: &'static str,
    pub intro: &'static str,
    pub takeaway: &'static str,
    pub lines: &'static [&'static str],
}

/// Which compute backend a demo targets. `Cpu` demos run anywhere,
/// including the in-browser WASM interpreter on the public live
/// demo. `Mlx` (Apple GPU) and `Cuda` (NVIDIA/Linux GPU) are
/// SEPARATE, connect-only groups -- each needs a `mlpl-serve` with
/// the matching device peer and never runs on the public live demo.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    Cpu,
    Mlx,
    Cuda,
}

/// A demo's runtime-requirement tier. Demos absent from
/// [`DEMO_CAPABILITIES`] default to [`Capability::CPU_LIVE`] --
/// runnable everywhere.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Capability {
    /// True when the demo needs a connected `mlpl-serve`; such
    /// demos render visible-but-not-runnable on the public live
    /// demo (which has no server).
    pub requires_connect: bool,
    pub device: Device,
}

impl Capability {
    /// The default tier: CPU, runnable on the public live demo.
    pub const CPU_LIVE: Self = Self {
        requires_connect: false,
        device: Device::Cpu,
    };
}

/// Per-demo capability overrides, keyed by `Demo::name`. Anything
/// absent is [`Capability::CPU_LIVE`]; only connect/GPU demos need
/// an entry. MLX and CUDA are deliberately distinct devices so the
/// UI can group and gate them separately.
pub const DEMO_CAPABILITIES: &[(&str, Capability)] = &[
    (
        "Ask Ollama (contextual)",
        Capability {
            requires_connect: true,
            device: Device::Cpu,
        },
    ),
    (
        "MLX LoRA fine-tune",
        Capability {
            requires_connect: true,
            device: Device::Mlx,
        },
    ),
    (
        "MLX tic-tac-toe fine-tune",
        Capability {
            requires_connect: true,
            device: Device::Mlx,
        },
    ),
    (
        "CUDA LoRA fine-tune",
        Capability {
            requires_connect: true,
            device: Device::Cuda,
        },
    ),
    (
        "CUDA tic-tac-toe fine-tune",
        Capability {
            requires_connect: true,
            device: Device::Cuda,
        },
    ),
];

/// Whether a demo with capability `cap` should be DISABLED
/// (visible-but-not-runnable) in the UI, given the page connection
/// state and the connected peer's device set (from `GET /v1/devices`).
///
/// Gating keys off the peer's REAL capability, not a static guess: a
/// connect demo is runnable only when `connected` AND the peer offers
/// the demo's device. Every peer has `cpu`, so a cpu connect demo
/// needs only a connection; an `mlx`/`cuda` demo needs that GPU in the
/// peer's set -- so a CUDA demo lights up against a CUDA peer but stays
/// disabled against an MLX-only peer (and vice versa). Non-connect
/// (live) demos are always runnable.
#[must_use]
pub fn demo_disabled(cap: &Capability, connected: bool, peer_devices: &[Device]) -> bool {
    if !cap.requires_connect {
        return false;
    }
    let peer_offers = cap.device == Device::Cpu || peer_devices.contains(&cap.device);
    !(connected && peer_offers)
}

/// The capability tier for `demo_name`, defaulting to
/// [`Capability::CPU_LIVE`] when the demo has no override.
pub fn capability_for(demo_name: &str) -> Capability {
    DEMO_CAPABILITIES
        .iter()
        .find(|(n, _)| *n == demo_name)
        .map_or(Capability::CPU_LIVE, |(_, c)| *c)
}

/// Per-demo companion literate HTML, keyed by `Demo::name`. The value
/// is the file under the deployed `literate/` directory (see
/// `examples/literate/*.org` -> published `.html`, bundled into
/// `pages/literate/`). A demo with an entry shows a "literate
/// walkthrough" link in its intro -- especially useful for
/// connect-only demos that the public live demo cannot run.
pub const LITERATE_DOCS: &[(&str, &str)] = &[
    ("Basics", "basics.html"),
    ("MLX LoRA fine-tune", "mlx-lora-finetune.html"),
    ("MLX tic-tac-toe fine-tune", "tictactoe-finetune.html"),
    ("CUDA LoRA fine-tune", "cuda-lora-finetune.html"),
    ("CUDA tic-tac-toe fine-tune", "cuda-tictactoe.html"),
];

/// The companion literate HTML filename for `demo_name`, if any.
pub fn literate_for(demo_name: &str) -> Option<&'static str> {
    LITERATE_DOCS
        .iter()
        .find(|(n, _)| *n == demo_name)
        .map(|(_, f)| *f)
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
        demo: "Watch a Model Learn (overfitting)",
        line_idx: 9,
        heading: "Training in six short bursts (~10s total on the CPU)",
        body: "The model trains 25 Adam steps at a time, six times over, scoring both the 30-point training set and the 200-point validation set after each burst. The page repaints between bursts -- the train_val_curve renders at the halfway mark and again at the end so you can watch the green dashed (train) and peach dotted (validation) lines pull apart. The widening gap is overfitting.",
    },
    ProgressNote {
        demo: "Taming Overfitting: Weight Decay",
        line_idx: 17,
        heading: "Training in six bursts with an L2 penalty (~10s on the CPU)",
        body: "Same over-capacity net and tiny noisy data as the overfitting demo, but the loss now adds lam * sum(W*W), which pushes the weights toward zero. The page repaints between bursts; the train_val_curve renders at the halfway mark and the end. Watch the validation curve stay near the training curve instead of peeling away.",
    },
    ProgressNote {
        demo: "Watch a Model Generalize (no overfitting)",
        line_idx: 9,
        heading: "Training in five short bursts (~10s total on the CPU)",
        body: "Same loop as the overfitting demo, but with a much larger training set (200 points) and a right-sized network. Watch the green dashed (train) and peach dotted (validation) curves fall together and stay close -- the small, stable gap is healthy generalization. Validation accuracy prints at the end.",
    },
    ProgressNote {
        demo: "CUDA tic-tac-toe fine-tune",
        line_idx: 13,
        heading: "Generating the self-play dataset on the CPU (~4s)",
        body: "This line plays 8 tic-tac-toe games where O uses recursive alpha-beta minimax to pick the optimal move -- pure game-tree search in the interpreter, so it runs on the CPU and the GPU stays idle here. That is expected: the GPU's turn is the next step (the LoRA fine-tune). Watch nvtop light up on the train line, not this one.",
    },
    ProgressNote {
        demo: "CUDA tic-tac-toe fine-tune",
        line_idx: 17,
        heading: "Fine-tuning the policy on the NVIDIA GPU (~10s)",
        body: "6000 Adam steps over the board-policy MLP, each a forward + backward + optimizer update run on the GPU via candle (device(\"cuda\")). This is the GPU-bound phase -- nvtop should show sustained utilization. The same train on the CPU is ~50x slower.",
    },
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
