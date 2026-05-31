//! Step 010: recognize the demo LoRA architecture and pull out what the
//! MLX fast path needs -- the frozen-weight + adapter param NAMES (from
//! the `ModelSpec`) and the X/Y token arrays (from the loss expr). Any
//! shape that does not match returns `None`, so `eval_adam` falls back
//! to the CPU tape path for every other model.

use crate::env::Environment;
use mlpl_array::DenseArray;
use mlpl_eval_core::ModelSpec;
use mlpl_parser::Expr;

/// Param names recovered from the demo model. Everything but the head
/// adapters (`head_a`, `head_b_adapter`) is a frozen base weight.
pub(crate) struct DemoLayout {
    pub embed_table: String,
    pub vocab: usize,
    pub wq: String,
    pub wk: String,
    pub wv: String,
    pub wo: String,
    pub head_w: String,
    pub head_b: String,
    pub head_a: String,
    pub head_b_adapter: String,
    pub alpha: f64,
    pub rank: usize,
}

// Attention projection names from a `Residual(Chain[RmsNorm, causal
// single-head Attention])` block, or None if the shape differs.
fn attn_names(layer: &ModelSpec) -> Option<(String, String, String, String)> {
    let ModelSpec::Residual(inner) = layer else {
        return None;
    };
    let ModelSpec::Chain(c) = inner.as_ref() else {
        return None;
    };
    if c.len() != 2 || !matches!(c[0], ModelSpec::RmsNorm { .. }) {
        return None;
    }
    match &c[1] {
        ModelSpec::Attention {
            wq,
            wk,
            wv,
            wo,
            heads: 1,
            causal: true,
            ..
        } => Some((wq.clone(), wk.clone(), wv.clone(), wo.clone())),
        _ => None,
    }
}

/// Recognize `Chain[Embedding, Residual(RmsNorm, causal Attention h=1),
/// RmsNorm, LinearLora]` and return its param names; None otherwise.
pub(crate) fn demo_layout(model_arg: &Expr, env: &Environment) -> Option<DemoLayout> {
    let Expr::Ident(name, _) = model_arg else {
        return None;
    };
    let ModelSpec::Chain(ls) = env.get_model(name)? else {
        return None;
    };
    if ls.len() != 4 || !matches!(ls[2], ModelSpec::RmsNorm { .. }) {
        return None;
    }
    let ModelSpec::Embedding { table, vocab, .. } = &ls[0] else {
        return None;
    };
    let (wq, wk, wv, wo) = attn_names(&ls[1])?;
    let ModelSpec::LinearLora {
        w,
        b,
        a,
        b_adapter,
        alpha,
        rank,
        ..
    } = &ls[3]
    else {
        return None;
    };
    Some(DemoLayout {
        embed_table: table.clone(),
        vocab: *vocab,
        wq,
        wk,
        wv,
        wo,
        head_w: w.clone(),
        head_b: b.clone(),
        head_a: a.clone(),
        head_b_adapter: b_adapter.clone(),
        alpha: *alpha,
        rank: *rank,
    })
}

/// Extract `(X, Y)` token arrays from `cross_entropy(apply(model, X), Y)`.
pub(crate) fn extract_xy(loss: &Expr, env: &mut Environment) -> Option<(DenseArray, DenseArray)> {
    let Expr::FnCall { name, args, .. } = loss else {
        return None;
    };
    if name != "cross_entropy" || args.len() != 2 {
        return None;
    }
    let Expr::FnCall {
        name: inner,
        args: iargs,
        ..
    } = &args[0]
    else {
        return None;
    };
    if inner != "apply" || iargs.len() != 2 {
        return None;
    }
    Some((eval_tokens(&iargs[1], env)?, eval_tokens(&args[1], env)?))
}

fn eval_tokens(e: &Expr, env: &mut Environment) -> Option<DenseArray> {
    crate::eval::eval_expr(e, env, &mut None)
        .ok()?
        .into_array()
        .ok()
}
