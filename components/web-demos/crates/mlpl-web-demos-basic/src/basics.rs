//! Saga 33 step 001: foundational demos (arithmetic, math,
//! arrays, viz helpers) extracted from `demos.rs`. Pure
//! compile-time data; consumed by `demos::DEMOS`.

use mlpl_web_demos_types::Demo;

pub const ANALYSIS_HELPERS: Demo = Demo {
    category: "Basics",
    name: "Analysis Helpers",
    intro: "Tour of the high-level viz helpers: histogram, labeled scatter, loss curve, confusion matrix, and a 2D decision-boundary surface. Each returns an SVG that renders inline. Use these as building blocks when you want one line of code to answer one question about your data or model.",
    takeaway: "Six one-liners, six labeled plots. Every helper under 'analysis' in the docs takes arrays you already have and returns an SVG -- no separate plotting library, no config blocks.",
    lines: &[
        "hist([1, 2, 2, 3, 3, 3, 4, 4, 5], 5)                                # histogram with 5 equal-width bins",
        "scatter_labeled([[0,0],[1,1],[0,1],[1,0]], [0, 0, 1, 1])            # 2D points colored by per-row label",
        "loss_curve([5, 3, 2, 1.5, 1.0, 0.7, 0.5, 0.4, 0.3, 0.25])           # training-loss line plot",
        "confusion_matrix([0, 1, 2, 1, 0], [0, 1, 1, 1, 0])                  # KxK predicted-vs-actual heatmap",
        "gx = grid([0, 1, 0, 1], 20)                                         # 20x20 (x, y) input-space grid",
        "boundary_2d(reshape(range(400), [400]) / 400, [20, 20], [[0,0],[1,1]], [0, 1])  # synthetic gradient surface as boundary",
    ],
};

pub const BASICS: Demo = Demo {
    category: "Basics",
    name: "Basics",
    intro: "The smallest possible MLPL tour: scalar arithmetic, elementwise array arithmetic with broadcasting, variable binding, and unary negation. If this makes sense, you can read the rest of the demos.",
    takeaway: "Operators apply elementwise; scalars broadcast; variables persist across REPL lines. That's the substrate every other demo builds on.",
    lines: &[
        "1 + 2                       # scalar addition",
        "3 * 4                       # scalar multiplication",
        "10 / 3                      # all numbers are f64; integer division does not exist",
        "[1, 2, 3] + [4, 5, 6]       # elementwise vector add",
        "[1, 2, 3] * 10              # scalar broadcasts across the vector",
        "x = [10, 20, 30]            # bind a vector to x",
        "y = x + 1                   # x is unchanged; y is a new vector",
        "y                           # echo y to see [11, 21, 31]",
        "-[1, 2, 3]                  # unary negation",
    ],
};

pub const LOSS_CURVE: Demo = Demo {
    category: "Basics",
    name: "Loss Curve",
    intro: "Sweep a single weight across 25 values, compute the MSE loss against a linear target at each one, and plot the result. No training -- just the shape of the loss landscape.",
    takeaway: "A smooth parabolic curve with a clear minimum near the true weight. This is what gradient descent is walking down when you train; seeing the bowl makes the 'minimize the loss' story tangible.",
    lines: &[
        "x = [0, 1, 2, 3, 4]                                # input x-values",
        "y = [0, 2, 4, 6, 8]                                # target y-values (slope 2)",
        "ws = range(25) / 4 - 1                              # 25 candidate weights from -1 to 5",
        "WS = reshape(ws, [25, 1])                          # column vector of weights",
        "preds = matmul(WS, reshape(x, [1, 5]))             # prediction for each weight x each x-value",
        "YS = matmul(ones([25, 1]), reshape(y, [1, 5]))     # broadcast targets across the 25 weights",
        "errs = preds - YS                                  # residuals",
        "losses = reduce_add(errs * errs, 1) / 5            # MSE per weight",
        "svg(losses, \"line\")                               # the loss bowl",
    ],
};

pub const MATH_FUNCTIONS: Demo = Demo {
    category: "Basics",
    name: "Math Functions",
    intro: "The scalar and elementwise math primitives MLPL inherits from NumPy/APL conventions: exp, log, sqrt, abs, pow, sigmoid, tanh. Works on scalars and arrays uniformly.",
    takeaway: "Every primitive broadcasts over array inputs without a loop. If you know these names from NumPy you already know MLPL's math surface.",
    lines: &[
        "exp(0)                            # exp on a scalar -> 1.0",
        "exp([0, 1, 2])                    # exp broadcasts elementwise",
        "log(exp(1))                       # log inverse of exp",
        "sqrt([4, 9, 16, 25])              # elementwise sqrt",
        "abs([-3, 0, 5])                   # elementwise absolute value",
        "pow([2, 3, 4], 2)                 # elementwise squaring",
        "sigmoid(0)                        # logistic at zero is 0.5",
        "sigmoid([-2, -1, 0, 1, 2])        # full S-curve sample",
        "tanh_fn([-1, 0, 1])               # tanh on a small vector",
    ],
};

pub const MATRIX_OPS: Demo = Demo {
    category: "Basics",
    name: "Matrix Ops",
    intro: "Build a 3x4 matrix from range, transpose it, read its shape and rank, and sum along both axes. The axis argument to reduce_add is how you go from a 2D tensor to a row-sum or column-sum vector.",
    takeaway: "reshape moves between flat and multi-dimensional views without copying; transpose swaps axes; reduce_add with an axis drops that axis. This is the APL half of MLPL -- shape is first-class and cheap to manipulate.",
    lines: &[
        "x = range(12)                       # flat 0..11 vector",
        "m = reshape(x, [3, 4])             # reshape to a 3x4 matrix",
        "m                                  # display the matrix",
        "transpose(m)                       # swap to 4x3",
        "shape(m)                           # the dimension vector [3, 4]",
        "rank(m)                            # number of dimensions (2)",
        "reduce_add(m, 0)                   # column sums (length 4)",
        "reduce_add(m, 1)                   # row sums (length 3)",
    ],
};

pub const WORKSPACE_INTROSPECTION: Demo = Demo {
    category: "Basics",
    name: "Workspace Introspection",
    intro: "Tour of the REPL's introspection commands: :version, :wsid (workspace ID summary), :vars, :describe, :models, :fns, :experiments. Also shows how the axis-label annotation syntax (M : [batch, feat] = ...) shows up in :vars and :describe output. When connected to an mlpl-serve (?connect=<url>) you also get the connect-mode introspectors -- :status (live CPU/GPU/RAM), :connect list (the server's Ollama LLMs for :ask), and :ask itself. NOTE the difference: :models lists the MLPL model objects YOU built here; :connect list lists the server's LLMs for :ask -- they are not the same. To avoid repeating all of this, the demo does NOT run :introspect -- that one command bundles every no-arg inspector (and names the connect-mode ones) into one markdown-headered dump; run it yourself to see the whole snapshot in a single scroll.",
    takeaway: "You can always ask the REPL what's in your session. :describe on a variable prints shape + labels + a preview; on a model, the layer tree; on a builtin, the signature and one-line doc. :experiments shows every tracked run. Don't confuse :models (your workspace's MLPL models) with :connect list (the connected server's Ollama LLMs for :ask). When you want the lot in one shot rather than command-by-command, run :introspect (saga 33 step 037d) -- it prints every no-arg inspector under `## :<topic>` headers with a trailing connect-mode section.",
    lines: &[
        ":version                                                            # build banner: version + arch + commit + timestamp",
        ":wsid                                                                # workspace summary (var/param/model counts)",
        "x = 42                                                               # bind a scalar",
        "v = range(5)                                                          # bind a 5-vector",
        "M : [batch, feat] = reshape(range(6), [2, 3])                         # bind a labeled-axis matrix",
        ":vars                                                                # list all bound variables with shape + tag",
        ":describe v                                                          # shape + values preview for a variable",
        "mdl = chain(linear(2, 4, 11), relu_layer(), linear(4, 2, 12))        # bind a model",
        ":models                                                              # list bound models with layer trees",
        ":describe mdl                                                        # the layer tree for mdl",
        "tok = train_bpe(\"abababab\", 260, 0)                                  # bind a tokenizer",
        ":describe tok                                                        # tokenizer info",
        "W = param[3, 2]                                                      # declare a trainable parameter",
        ":vars                                                                # see the [param] tag on W",
        ":wsid                                                                # parameter count went up",
        "experiment \"workspace_demo\" { loss_metric = 0.25; accuracy_metric = 0.94 }  # capture two metrics into the experiment registry",
        ":experiments                                                         # list every captured run",
        ":describe matmul                                                     # signature + one-line doc for a builtin",
        ":fns                                                                 # list user-defined functions (none yet)",
        ":status                                                              # connect mode: backend devices + live CPU/GPU/RAM (needs ?connect=)",
        ":connect list                                                        # connect mode: the server's Ollama LLMs for :ask -- NOT the :models above",
        ":ask \"what have I built in this workspace so far?\"",
    ],
};

pub const VISUALIZATIONS: Demo = Demo {
    category: "Basics",
    name: "Visualizations",
    intro: "The four primitive svg() types -- scatter, line, bar, heatmap -- each in one line. Rendered inline; a download button next to each SVG saves it as a file.",
    takeaway: "Every plot is one call with a data array and a type string. There is no plotting API to learn beyond 'pass the right shape.'",
    lines: &[
        "svg([[0,0],[1,1],[2,4],[3,9],[4,16]], \"scatter\")    # Nx2 -> circle per row",
        "svg([1, 3, 2, 5, 4, 6], \"line\")                      # vector -> polyline",
        "svg([3, 1, 4, 1, 5, 9, 2, 6], \"bar\")                 # vector -> bar chart",
        "svg(reshape(range(25), [5, 5]), \"heatmap\")            # MxN -> viridis grid",
    ],
};

pub const UPLOAD_INSPECT: Demo = Demo {
    category: "Basics",
    name: "Upload & Inspect Image",
    intro: "How to bring your own image into MLPL. Type :upload x in the REPL \
            to open your device's file picker. The browser decodes and resizes \
            the photo to 64x64 and binds it as a Result -- Ok({pixels, h, w}) \
            on success, Err(message) on cancel or decode failure. This demo \
            simulates the workflow with synthetic pixel data so you can see \
            the inspection pattern before trying :upload yourself.",
    takeaway: "The pattern is always: :upload name, then is_ok(name) to check, \
               then unwrap(name).pixels to get the tensor. From there, shape, \
               mean, min, max tell you what you have. svg(pixels, \"gallery\") \
               renders it as a thumbnail grid. Try :upload x now and repeat \
               these steps on your own photo.",
    lines: &[
        "# Simulate an uploaded 64x64 RGB image (batch of 1)",
        "pixels = reshape(random(42, [12288]), [1, 3, 64, 64])",
        "shape(pixels)",
        "# Inspect the pixel values",
        "mean(pixels)",
        "reduce(:min, pixels)",
        "reduce(:max, pixels)",
        "# Display as a gallery thumbnail",
        "svg(pixels, \"gallery\")",
        "# Histogram of pixel intensities",
        "hist(reshape(pixels, [12288]), 20)",
    ],
};

pub const BPE_ATTENTION_LABELS: Demo = Demo {
    category: "Basics",
    name: "BPE-Labeled Attention Pattern",
    intro: "Saga BPE-2: tokenize a short text with train_bpe + apply_tokenizer, decode each token id back to its byte string via decode_each, then run attention on the embedded sequence. Click the `A` sculpture (rank-3 [heads, T, T] attention weights) and the heatmap axes label rows + columns with the actual BPE pieces -- not integer indices -- because `labels = decode_each(...)` rode through the new ShapeInfo.string_list channel into the next attention sculpture.",
    takeaway: "Per-token visualization needs the per-token labels alongside the data. decode_each is the BPE-side counterpart to decode (which collapses the whole id sequence into one string); the viz plumbing in ShapeInfo carries the resulting StrList forward so the next attention heatmap renders with real tokens on its axes. Untrained: the heatmap is randn noise; the point here is that the labels flow, not that the model knows anything.",
    lines: &[
        "# Saga BPE-2: end-to-end BPE labels on an attention heatmap.",
        "corpus = \"the quick brown fox jumps over the lazy dog\"      # 9 ASCII words",
        "tok    = train_bpe(corpus, 80, 0)                            # 80 merges -> multi-char tokens emerge",
        "ids    = apply_tokenizer(tok, corpus)                        # rank-1 token ids",
        "labels = decode_each(tok, ids)                                # StrList: per-token byte strings",
        "# Above `labels` rides through ShapeInfo.string_list and is",
        "# attached to the NEXT attention sculpture as its axis labels.",
        "d = 16                                                         # embedding dim",
        "V = 300                                                        # max vocab id + slack (BPE vocab = 256 + merges)",
        "emb = embed(V, d, 1)                                          # embedding layer",
        "seq = apply(emb, ids)                                         # [N, d] embedded sequence",
        "# Two-head attention, max sequence length 64 (well above the corpus length).",
        "mdl    = attention(d, 2, 64)",
        "A      = attention_weights(mdl, seq)                          # [2, N, N]",
        "svg(A, \"heatmap_grid\")                                       # peek before clicking",
        "# Click the `A` sculpture in the 3D stage -- the dialog should",
        "# show real BPE pieces on the axes instead of 0..N-1.",
    ],
};
