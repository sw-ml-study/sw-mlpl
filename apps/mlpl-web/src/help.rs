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
  core       iota, shape, rank, reshape, transpose,
             reduce, reduce_add, reduce_mul, argmax, mean
  labels     label, relabel, reshape_labeled, labels
  linalg     dot, matmul
  math       exp, log, sqrt, abs, pow, sigmoid, tanh_fn
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
  embed      pairwise_sqdist, knn, tsne, pca, embed_table
  estimate   estimate_train, estimate_hypothetical, feasible,
             calibrate_device
  viz        svg, hist, scatter_labeled, loss_curve,
             confusion_matrix, boundary_2d
  llm        llm_call

Higher-order: reduce(:op, x[, axis]) where :op is one of
  :add  /  :+    sum
  :mul  /  :*    product
  :max           per-axis max
  :min           per-axis min
  :and  /  :or   boolean reductions over 0/1 masks

Commands (APL-inspired workspace introspection):
  :help                show this help
  :help <topic>        focused help: vars, models, fns,
                       builtins, describe, wsid
  :version             sw-MLPL version + target arch
  :vars                list bound variables with shape and tag
  :models              list bound models with layer structure
  :tokenizers          list bound tokenizer values
  :fns                 list user-defined functions (APL )FNS)
  :builtins            list built-in functions by category
  :describe <name>     describe a variable, model, tokenizer,
                       or built-in (with v0.19 typed header)
  :tags                list every binding's ValueTag (Saga 23)
  :untag <name>        clear a binding's auto-attached tag
  :wsid                workspace summary (APL )WSID)
  :experiments         list captured experiment runs
  :ask <prompt>        send prompt to a local Ollama LLM
  :clear               reset session (vars + models + state)";

pub fn help_text() -> String {
    HELP_TEXT.to_string()
}
