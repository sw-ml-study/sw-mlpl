use crate::types::{LearningPath, Step};

pub(super) const PATH_BUILD_A_TRANSFORMER_FROM_PRIMITIVES: LearningPath = LearningPath {
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
};

pub(super) const PATH_DATA___EXPLORATION: LearningPath = LearningPath {
    title: "Data & Exploration",
    blurb: "Before you model, explore. This path walks the data side of ML: creating arrays, uploading images, inspecting shapes, visualizing distributions, generating synthetic datasets, and preparing data for training. Every step produces a picture or a number -- no models, no gradients, just getting to know your data.",
    steps: &[
        Step::Note {
            title: "Why exploration matters",
            body: "Most ML failures are data failures. A model trained on skewed data learns skewed patterns. A model trained on the wrong scale diverges. Spending 10 minutes looking at histograms and scatter plots before training saves hours of debugging after. This path builds the habits.",
        },
        Step::Lesson {
            title: "Hello Numbers",
            why: "Scalars, operators, the REPL. The absolute starting point -- everything else is arrays of numbers.",
        },
        Step::Lesson {
            title: "Arrays",
            why: "Vectors and their shapes. range(n) generates a sequence; reshape changes the layout. Arrays are the container for every dataset.",
        },
        Step::Lesson {
            title: "Matrices",
            why: "Reshape, transpose, slicing with take. A dataset is a matrix: rows are samples, columns are features.",
        },
        Step::Demo {
            name: "Basics",
            why: "Scalar arithmetic, elementwise ops, broadcasting, variable binding. The five-minute tour.",
        },
        Step::Demo {
            name: "Math Functions",
            why: "exp, log, sqrt, abs, sin, cos, sigmoid, tanh. The elementwise toolkit you will use for feature engineering and activation functions.",
        },
        Step::Note {
            title: "Generating synthetic data",
            body: "MLPL ships several synthetic dataset generators: blobs(seed, n, centers) for Gaussian clusters, moons(seed, n, noise) for two interleaving arcs, circles(seed, n, noise) for concentric rings, and random/randn for uniform/normal noise. Each returns a matrix ready for plotting or training. Seeded for reproducibility.",
        },
        Step::Demo {
            name: "Matrix Ops",
            why: "Reshape, transpose, matmul, dot. The shape-manipulation toolkit for arranging data into the format a model expects.",
        },
        Step::Lesson {
            title: "Visualizing Data",
            why: "svg(data, type) renders inline: line, bar, heatmap, scatter. One function, many views.",
        },
        Step::Demo {
            name: "Visualizations",
            why: "Line plots, bar charts, heatmaps in one line each. The visual vocabulary for data exploration.",
        },
        Step::Demo {
            name: "Analysis Helpers",
            why: "hist, scatter_labeled, loss_curve, confusion_matrix, boundary_2d. Higher-level plots that answer specific questions about your data or model.",
        },
        Step::Demo {
            name: "Upload & Inspect Image",
            why: "Bring your own data: :upload, check is_ok, inspect shape/mean/min/max, render with svg gallery, histogram of pixel intensities.",
        },
        Step::Glossary {
            term: ":upload (REPL command)",
            why: "The browser file picker: pick a photo, get a Result with pixels, height, width.",
        },
        Step::Lesson {
            title: "Loading Data",
            why: "load(path) reads CSV or text files. The terminal REPL needs --data-dir; the web playground has load_preloaded for bundled datasets.",
        },
        Step::Lesson {
            title: "Named Axes",
            why: "label(x, names) attaches semantic names to dimensions. 'batch', 'features', 'time' -- makes shapes self-documenting.",
        },
        Step::Note {
            title: "From exploration to modeling",
            body: "You now know how to create, load, inspect, and visualize data in MLPL. The next step is modeling: the 'Zero to LLM' path starts with logistic regression and builds to transformers. The 'Architecture Zoo' path surveys CNN, RNN, GAN, and attention side by side. Pick the one that matches your curiosity.",
        },
    ],
};

pub(super) const PATH_DIMENSIONALITY_REDUCTION: LearningPath = LearningPath {
    title: "Dimensionality reduction",
    blurb: "When the data lives in 50 dimensions but the screen has 2: pick a projection. PCA (linear, fast), t-SNE (local-only, dramatic), UMAP (local + global, the modern default). Six tutorial lessons in dependency order, three side-by-side demos at the end.",
    steps: &[
        Step::Lesson {
            title: "Why reduce dimensions?",
            why: "Concept-first motivation: the manifold hypothesis, the curse of dimensionality, why a screen has two axes but a learned embedding has 768. Frame the whole path.",
        },
        Step::Glossary {
            term: "Dimensionality reduction",
            why: "Reference card for the rest of the path: linear vs manifold methods, what each preserves, the MLPL builtins.",
        },
        Step::Demo {
            name: "PCA",
            why: "The cheap linear baseline. Power iteration on the covariance matrix finds the top axis of variance; project the points onto it. Linear, fast, deterministic.",
        },
        Step::Lesson {
            title: "PCA: the linear baseline",
            why: "Goes beyond the demo: pca() vs pca_components() vs pca_variance_explained(), the loadings-vs-projections distinction, when PCA's linear assumption is enough.",
        },
        Step::Glossary {
            term: "PCA (Principal Component Analysis)",
            why: "What PCA actually does (eigenvectors of the covariance matrix), what it misses (non-linear structure).",
        },
        Step::Demo {
            name: "PCA 3D (interactive)",
            why: "Same dataset, but project to 3 components and view in an interactive Plotly viewer. Drag/rotate confirms that well-separated 5-D clusters stay separated along every axis -- harder to fake than a single 2-D shot.",
        },
        Step::Demo {
            name: "PCA loadings (critical dimensions)",
            why: "Switch from 'where did the points go?' to 'which input dimensions matter?'. The critical-dimensions heatmap shows which features each PC is built from.",
        },
        Step::Lesson {
            title: "SNE: the very-slow ancestor",
            why: "t-SNE's predecessor (Hinton + Roweis 2002). Two failure modes -- asymmetric KL and the crowding problem -- set up exactly the two fixes t-SNE makes.",
        },
        Step::Lesson {
            title: "t-SNE: a peek at nonlinear methods",
            why: "How van der Maaten + Hinton fixed SNE: symmetric P + Student-t low-D affinity. Plus the 'cluster shape is meaningful, distance between clusters is not' caveat.",
        },
        Step::Glossary {
            term: "t-SNE",
            why: "Reference card: perplexity, KL, Student-t, why global distance is noise.",
        },
        Step::Lesson {
            title: "UMAP: the modern default",
            why: "Headline lesson. Riemannian-geometry framing, fuzzy simplicial sets, cross-entropy + negative sampling. Why UMAP preserves both local AND global structure where t-SNE preserves only local.",
        },
        Step::Glossary {
            term: "UMAP",
            why: "Reference card for the lesson: the math vocabulary in one place.",
        },
        Step::Demo {
            name: "UMAP vs PCA",
            why: "Two-moons embedded in 5-D. PCA reads the linear projection; UMAP reads the local k-NN graph. Both recover the moon arcs but via different recipes.",
        },
        Step::Demo {
            name: "UMAP vs t-SNE",
            why: "Three clusters where C is 5x farther than A is from B. t-SNE inflates every cluster to similar size; UMAP preserves the relative distance. This is the case the milestone is built around.",
        },
        Step::Glossary {
            term: "Multidimensional Scaling",
            why: "What MDS preserves (pairwise distances) and how it differs from PCA (variance directions) and t-SNE/UMAP (local neighborhoods). Background for the next demo.",
        },
        Step::Glossary {
            term: "Johnson-Lindenstrauss Lemma",
            why: "The sanity-baseline argument: a Gaussian random matrix preserves pairwise distances within (1 +- eps) for modest k. If a learned method does not beat random projection, the learned features are not adding signal.",
        },
        Step::Demo {
            name: "Dim-reduction zoo",
            why: "Same dataset, FIVE side-by-side projections (PCA / t-SNE / UMAP / MDS / random projection). The fastest way to internalize what each method emphasizes -- variance directions vs local neighborhoods vs pairwise distances vs the JL sanity baseline.",
        },
        Step::Lesson {
            title: "Reading a critical-dimensions heatmap",
            why: "Viz literacy. How to read the [k, D] critical-dimensions heatmap for PCA loadings; same conventions apply to upcoming permutation-sensitivity heatmaps for t-SNE / UMAP.",
        },
        Step::Note {
            title: "What's next",
            body: "The dim-reduction milestone is feature-complete after step 041 (MDS + random projection landed and joined the zoo). The remaining headroom is permutation-sensitivity heatmaps for t-SNE / UMAP (referenced in the 'Reading a critical-dimensions heatmap' lesson) -- those need a small helper builtin and ship as a follow-on saga. Track the milestone status in `docs/milestone-dimensionality-reduction.md`.",
        },
    ],
};

pub(super) const PATH_REPL_TO_SCRIPT: LearningPath = LearningPath {
    title: "REPL to Script",
    blurb: "Graduate from one-line REPL exploration to multi-line scripts to saved .mlpl files. Learn the editor, the save/load workflow, terminal script mode, user-defined functions, and script arguments. Eight steps; assumes you have run a few REPL expressions already.",
    steps: &[
        Step::Note {
            title: "From exploration to automation",
            body: "The REPL is for exploring: try an expression, see the result, adjust. Once you have something that works, you want to save it, run it again, and share it. This path walks the progression from interactive one-liners to reusable scripts.",
        },
        Step::Lesson {
            title: "Hello Numbers",
            why: "The REPL basics: type an expression, press Enter, see the result. Variables persist across lines. :vars shows what is bound.",
        },
        Step::Lesson {
            title: "Variables",
            why: "Name your intermediate values. x = range(10) binds an array; x persists until :clear. Variable names are your working memory.",
        },
        Step::Demo {
            name: "Workspace Introspection",
            why: "The REPL's self-awareness: :vars, :describe, :models, :fns, :wsid. Know what is in your session before you save it.",
        },
        Step::Note {
            title: "The Editor tab",
            body: "Click the Editor tab to get a multi-line text area. Type or paste several lines of MLPL, then press Ctrl+Enter (or the Run button) to execute them all. Output appears in the REPL pane below. The editor is a scratchpad -- it does not save automatically.",
        },
        Step::Note {
            title: "Saving and loading scripts",
            body: "The Save button downloads your editor content as a .mlpl file. The Load button opens a file picker to load a .mlpl file into the editor. The browser does not persist state between sessions -- save early, save often. You can also copy/paste between the editor and any text editor on your machine.",
        },
        Step::Demo {
            name: "User-Defined Functions",
            why: "def u:name(args) { body } defines a reusable function. This is how scripts become libraries: define your functions at the top, call them below.",
        },
        Step::Note {
            title: "Running scripts from the terminal",
            body: "The terminal REPL runs .mlpl files directly: mlpl-repl -f my_script.mlpl. Use -f (not stdin piping) because piping splits multi-line blocks like repeat {} across lines. Script arguments: mlpl-repl -f script.mlpl -- arg1 arg2. Inside the script, args() returns a string list and list_get(args(), 0) extracts one argument.",
        },
        Step::Note {
            title: "What comes next",
            body: "You now know the full REPL-to-script workflow: explore interactively, draft in the editor, save as .mlpl, run from the terminal with arguments. User-defined functions (def u:name) let you build reusable libraries. The Architecture Zoo and Zero to LLM paths show what to build with these tools.",
        },
    ],
};

pub(super) const PATH_ZERO_TO_LLM: LearningPath = LearningPath {
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
};

pub(super) const PATH_HOW_DOES_ML_WORK__START_HERE: LearningPath = LearningPath {
    title: "How does ML work? (start here)",
    blurb: "The single best place to begin. No prior ML needed -- if you can read a little arithmetic, you can follow this. We build the one idea under everything (adjust some numbers to make predictions less wrong), watch it happen on screen, see how it goes wrong, and finish at a model that generates text. Ten short steps; every one shows a picture or a number.",
    steps: &[
        Step::Note {
            title: "What you are about to learn",
            body: "Machine learning sounds mysterious, but the core loop is simple: a model is just a bag of numbers (its 'weights'); you measure how wrong its predictions are (the 'loss'); and you nudge the numbers to make the loss smaller. Repeat a few thousand times and the numbers become a model that recognizes images or writes text. This path shows that loop directly -- you will literally watch a model roll downhill toward a good answer.",
        },
        Step::Lesson {
            title: "Hello Numbers",
            why: "Start at zero: scalars, operators, the REPL. Everything in ML is arithmetic on arrays of numbers, so we begin with the numbers.",
        },
        Step::Demo {
            name: "Basics",
            why: "The five-minute tour -- arrays, elementwise math, broadcasting. This is the raw material every model is built from.",
        },
        Step::Demo {
            name: "How Gradient Descent Works",
            why: "The heart of the whole field, made visible. Fit a line by drawing the entire loss surface and watching the optimizer walk downhill into the valley. After this, 'training' is no longer a black box.",
        },
        Step::Glossary {
            term: "Gradient descent",
            why: "The name for that downhill walk. The gradient is the slope; you step against it to reduce the loss.",
        },
        Step::Glossary {
            term: "Loss Landscape",
            why: "The surface you just watched. Training is search for its lowest point.",
        },
        Step::Demo {
            name: "Watch a Model Learn (overfitting)",
            why: "Now a real (over-powered) network on noisy data. Training and validation loss are plotted together so you see the model start to memorize noise -- the most important failure mode in ML.",
        },
        Step::Glossary {
            term: "Overfitting / Underfitting",
            why: "The gap you saw between the green and orange curves. Memorizing the training set is not the same as learning.",
        },
        Step::Demo {
            name: "Decision Boundary: XOR (with MLP)",
            why: "Why we stack layers: a single linear model cannot separate XOR, but one hidden layer + a nonlinearity bends the boundary until it can. The leap from 'line' to 'neural network'.",
        },
        Step::Demo {
            name: "Tiny LM Generate",
            why: "The payoff: the same downhill loop, scaled up to a 1-layer transformer that learns a tiny language and generates text token by token. From 'adjust two numbers' to 'write sentences' -- same idea throughout.",
        },
        Step::Note {
            title: "Where to go next",
            body: "You now have the whole loop end to end. To go deeper: 'Zero to LLM' rebuilds this story with more rigor; 'Build a transformer from primitives' opens up the attention machinery inside Tiny LM; and 'How models generate' (below) contrasts the different ways a model can produce new data.",
        },
    ],
};
