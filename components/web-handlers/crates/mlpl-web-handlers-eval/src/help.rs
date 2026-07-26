/// Static body of the `:help` panel in the web REPL. Pure
/// data; the only reason this lives in a function is to
/// satisfy the existing `output: String` signature in
/// `handlers.rs`.
const HELP_TEXT: &str = "sw-MLPL v0.19.0 -- Software Wrighter's Machine Learning
Programming Language

Syntax:
  42 / 1.5            scalar literal
  [1, 2, 3]           array literal
  \"text\"              string literal
  :foo / :+           builtin / operator reference
                       (e.g. reduce(:add, v) or
                        f = :max; reduce(f, v))
  x = expr            assignment
  x : [a, b] = expr   labeled-axis assignment
  a + b               arithmetic (+, -, *, /)
  func(args)          function call
  param[shape]        trainable parameter leaf
  tensor[shape]       non-trainable tape tensor
  repeat N { body }   loop N times
  train N { body }    training loop (binds step, last_losses)
  for row in X { }    iterate rows (binds last_rows)
  experiment \"n\" { }  capture _metric scalars + params
  device(\"t\") { }     dispatch body through device target
  # comment            to end of line

Built-in categories (see :help <topic> or :builtins for the
full list):
  core       range (iota), shape, rank, reshape, transpose,
             rotate, reduce, reduce_add, reduce_mul, argmax,
             mean
  labels     label, relabel, reshape_labeled, labels
  linalg     dot, matmul
  math       pi, e, exp, log, sqrt, abs, sin, cos, pow,
             sigmoid, tanh_fn
  ctors      zeros, ones, fill, grid, random, randn,
             blobs, moons, circles
  compare    gt, lt, eq
  ml         softmax, one_hot, cross_entropy, sample, top_k
  autograd   grad(expr, wrt)
  optim      momentum_sgd, adam, cosine_schedule,
             linear_warmup
  modeldsl   linear, chain, residual, rms_norm, attention,
             causal_attention, embed, sinusoidal_encoding,
             tanh_layer, relu_layer, softmax_layer, apply,
             params, freeze, unfreeze, lora, clone_model,
             perturb_params
  data       load, load_preloaded, shuffle, batch,
             batch_mask, split, val_split
  tokenize   tokenize_bytes, decode_bytes, train_bpe,
             apply_tokenizer, decode
  lm         shift_pairs_x, shift_pairs_y, last_row, concat,
             attention_weights, argtop_k, scatter
  experiment compare
  conv       conv2d, pool2d
  rnn        rnn_cell, lstm_cell
  loss       cross_entropy, perplexity
  embed      pairwise_sqdist, knn, tsne, pca, embed_table
  dimreduce  pca_components, pca_variance_explained,
             knn_graph, umap, mds, random_projection
  estimate   estimate_train, estimate_hypothetical, feasible,
             calibrate_device
  viz        svg, hist, scatter_labeled, scatter3d,
             loss_curve, confusion_matrix, boundary_2d
  llm        llm_call

User-defined functions:
  def u:name(args) { body }   define a function (u: prefix)
  return [expr]               early exit from a function body

Higher-order: reduce(:op, x[, axis]) where :op is one of
  :add  /  :+    sum
  :mul  /  :*    product
  :max           per-axis max
  :min           per-axis min
  :and  /  :or   boolean reductions over 0/1 masks

Commands (APL-inspired workspace introspection):
  :help                show this help (lists every command)
  :help <topic>        focused help: vars, models, fns,
                       builtins, describe, wsid
  :<cmd> --help        help for ONE command (e.g. :ask --help).
                       (bare args are command input: :ask help
                        asks the model \"help\")
  :version             sw-MLPL version + target arch
  :vars                list bound variables with shape and tag
  :models              MLPL models you built here (layer trees) --
                       NOT the server LLMs (see :connect list)
  :tokenizers          list bound tokenizer values
  :fns                 list user-defined functions (APL )FNS)
  :builtins            list built-in functions by category
  :describe <name>     describe a variable, model, tokenizer,
                       or built-in (with v0.19 typed header)
  :tags                list every binding's ValueTag
  :untag <name>        clear a binding's auto-attached tag
  :wsid                workspace summary (APL )WSID)
  :experiments         list captured experiment runs
  :status              backend self-test: connected backend(s),
                       devices + GPU type(s), live CPU/RAM/GPU/VRAM
  :status watch        sample the backend for a few seconds and draw
                       CPU/GPU/RAM/VRAM sparklines (on demand)
  :reset               cancel all in-flight work on the connected
                       backend (recover from a hung/slow demo)
  :ask <prompt>        send prompt (verbatim) to a connected Ollama LLM
  :connect list        the server's Ollama LLMs for :ask (not :models)
  :connect set <model> pick the Ollama model for :ask this session
  :introspect          run all no-arg inspectors at once
  :upload <name>       open file picker; bind chosen photo as
                       <name> = Ok({pixels, h, w}) or
                       Err(\"cancelled\") on dismiss (web only)
  :3d                  open 3D visualization stage (Ctrl+3)
  :2d                  close 3D visualization stage
  :3d reset            reset 3D camera to default position
  :clear               reset session (vars + models + state)";

pub fn help_text() -> String {
    HELP_TEXT.to_string()
}

/// Per-command help body. `--help` (or `-h`) is the REPL convention for
/// "explain this command" -- distinct from bare arguments, which are the
/// command's input (e.g. `:ask --help` shows this; `:ask help` sends
/// "help" to the Ollama model). Returns `None` for unknown commands.
fn command_help_body(cmd: &str) -> Option<&'static str> {
    Some(match cmd {
        ":ask" => {
            "Usage: :ask <question>\n  Send <question> verbatim to the connected Ollama model \
                   (connect mode only). No quotes needed -- everything after `:ask ` is the prompt, \
                   grounded with your recent REPL activity.\n  :ask --help  -> this help.   :ask help  -> ask the model \"help\".\n  Choose the model: :connect list (see installed) then :connect set <name>."
        }
        ":status" => {
            "Usage: :status [watch]\n  Report the connected backend(s): hostname, devices, GPU type(s), and live \
                      CPU/RAM/GPU/VRAM. 0 backends = local browser mode.\n  :status watch -> sample the backend for ~4s and draw on-demand sparklines."
        }
        ":reset" => {
            "Usage: :reset\n  Cancel ALL in-flight work on the connected backend (recover a hung/slow demo). \
                     Prompts (y/N) first. No-op in local mode."
        }
        ":connect" => {
            "Usage: :connect list | :connect set <model>\n  list        -> the server's installed Ollama models (current one marked).\n  set <model> -> pick the :ask model for this session."
        }
        ":history" => {
            "Usage: :history\n  List the recent REPL command lines (also given to :ask as context)."
        }
        other => return session_help_body(other),
    })
}

/// Second half of the per-command help table: session-lifecycle and
/// view commands. Split from `command_help_body` for the LOC budget.
fn session_help_body(cmd: &str) -> Option<&'static str> {
    Some(match cmd {
        ":clear" => {
            "Usage: :clear\n  Reset the session: clears variables, models, and 3D state.\n  Demos run in the SAME session until you reset -- so later demos and :ask can see earlier results."
        }
        ":help" => {
            "Usage: :help [topic]\n  :help -- this command list.  :<cmd> --help -- one command's help.\n  :help <topic> -- focused help (vars, models, fns, builtins, describe, wsid)."
        }
        ":3d" | ":2d" => {
            "Usage: :3d | :2d | :3d reset\n  :3d open the 3D stage (Ctrl+3); :2d close it; :3d reset re-centers the camera."
        }
        ":upload" => {
            "Usage: :upload <name>\n  Open a file picker and bind the chosen photo as <name> = Ok({pixels,h,w})."
        }
        _ => return None,
    })
}

/// If `line` is `:<cmd> --help` (or `-h`), return that command's help.
/// `None` otherwise (including a bare `:<cmd> help`, which is the
/// command's own input, not REPL help).
pub fn command_help(line: &str) -> Option<String> {
    let t = line.trim();
    let cmd = t
        .strip_suffix(" --help")
        .or_else(|| t.strip_suffix(" -h"))
        .map(str::trim)?;
    command_help_body(cmd).map(|body| format!("{cmd} -- {body}"))
}

/// Shown for `:ask` when there is no connected server (e.g. the
/// public live demo). `:ask` needs a server to reach an LLM, so we
/// give a clear notice instead of lexing the question as MLPL
/// (which errors on punctuation like `?`).
const ASK_NEEDS_SERVER: &str = "`:ask` is not available on the public demo -- it needs a connected mlpl-serve with Ollama running. Run `mlpl-serve` on a local machine (with `ollama serve`), then open this REPL with `?connect=<server-url>` appended to the page URL (e.g. `...?connect=http://host:6464`). The CUDA / MLX demos additionally need that server on a host with the matching GPU (Linux+NVIDIA for CUDA, Apple Silicon for MLX).";

/// Shown for `:status` when no server is connected. Connect mode
/// answers `:status` with a live backend report (devices + CPU/RAM/
/// GPU/VRAM); here there is no backend to probe.
const STATUS_LOCAL: &str = "Status: 0 backends connected -- local (browser) mode.\n  device  : cpu (browser WASM)\n  Live CPU/GPU/RAM/VRAM telemetry, :ask, and the CUDA/MLX demos need a\n  connected mlpl-serve. Start one (mlpl-serve --bind 0.0.0.0:6464\n  --auth required), then append ?connect=<server-url> to this page's URL\n  (e.g. ?connect=http://host:6464).";

/// Canned local-mode replies for REPL `:commands` that need no eval.
/// Connect-mode variants (`:status` probe, async `:connect list`,
/// `:reset` confirm) are handled upstream in `connect.rs`; reaching
/// here means no server is connected. `None` -> evaluate as MLPL.
pub(crate) fn command_reply(
    deps: &mlpl_web_handlers_upload::eval_deps::EvalDeps,
    trimmed: &str,
) -> Option<String> {
    if trimmed == ":ask" || trimmed.starts_with(":ask ") {
        return Some(ASK_NEEDS_SERVER.to_string());
    }
    if trimmed == ":connect" || trimmed.starts_with(":connect ") {
        let arg = trimmed.strip_prefix(":connect").unwrap_or("").trim();
        return Some(connect_command_text(arg));
    }
    match trimmed {
        ":help" => Some(help_text()),
        ":history" => Some(history_listing(deps)),
        ":status" => Some(STATUS_LOCAL.to_string()),
        ":reset" => {
            Some("Nothing to reset -- local browser mode, no connected server.".to_string())
        }
        _ => None,
    }
}

/// `:history` output -- the recent REPL command lines, so the
/// user (and the `:ask` LLM, which also receives this) can see
/// what has been run.
fn history_listing(deps: &mlpl_web_handlers_upload::eval_deps::EvalDeps) -> String {
    let lines: Vec<String> = deps
        .history
        .iter()
        .filter(|e| matches!(e.kind, mlpl_web_eval::state::EntryKind::Command))
        .map(|e| format!("mlpl> {}", e.input.trim()))
        .collect();
    if lines.is_empty() {
        "(no commands run yet)".to_string()
    } else {
        lines.join("\n")
    }
}

/// Text for `:connect <arg>` outside the async connect path. `set
/// <model>` selects the `:ask` model for the session (local/sync);
/// `list` / bare need a connected server (handled upstream when one is
/// connected, so here it means "not connected").
fn connect_command_text(arg: &str) -> String {
    if let Some(name) = arg.strip_prefix("set ").map(str::trim) {
        if name.is_empty() {
            return "usage: :connect set <model>   (see :connect list)".to_string();
        }
        if name.contains("embed") {
            return format!(
                "'{name}' is an embedding model (returns vectors, not text) -- it can't \
                 answer :ask. Pick a chat model instead (see :connect list)."
            );
        }
        #[cfg(target_arch = "wasm32")]
        mlpl_web_eval::ollama_fetch::set_selected_model(name);
        format!("ask model set to {name}")
    } else if arg == "list" || arg.is_empty() {
        format!(
            "`:connect list` needs a connected mlpl-serve to query its Ollama models. {ASK_NEEDS_SERVER}"
        )
    } else {
        format!("unknown :connect subcommand '{arg}' (try: list  |  set <model>)")
    }
}
