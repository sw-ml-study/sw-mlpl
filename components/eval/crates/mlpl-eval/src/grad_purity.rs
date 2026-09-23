//! Param-dependence analysis for grad's constant-fold fallback. The fold is
//! only sound for an expression that provably does not read any parameter's
//! VALUE; anything this analysis cannot see through counts as param-using, so a
//! genuinely non-differentiable subtree over a param errors instead of silently
//! dropping its gradient. A `u:` call is analyzed through its BODY as well as
//! its arguments: a body that reads a global param is never folded
//! (demo-decision-model Q5 -- folding it returned a silently wrong gradient).

use std::collections::{HashMap, HashSet};

use mlpl_autograd::Tensor;
use mlpl_parser::Expr;

use crate::env::Environment;
use crate::env_api::*;

/// Whether evaluating `expr` depends on the VALUE of any tracked parameter.
/// A parameter that appears only inside shape-metadata builtins (`shape`,
/// `rank`, `len`, `labels`) does not count -- those read structure, not values
/// -- so a shape-derived size is value-independent and safe to constant-fold.
/// A model identifier counts (its weights are params).
pub(crate) fn differentiably_uses_param(
    expr: &Expr,
    params: &HashMap<String, Tensor>,
    env: &Environment,
) -> bool {
    uses(expr, &Scope { params, env }, &mut HashSet::new())
}

struct Scope<'a> {
    params: &'a HashMap<String, Tensor>,
    env: &'a Environment,
}

/// Identifiers and calls carry the param semantics; every other form is
/// scanned through its structural children. A form with no known children is
/// conservatively param-using (never fold what we cannot prove constant).
fn uses(expr: &Expr, sc: &Scope, seen: &mut HashSet<String>) -> bool {
    match expr {
        Expr::Ident(n, _) => sc.params.contains_key(n) || sc.env.get_model(n).is_some(),
        Expr::FnCall { name, .. } if matches!(&**name, "shape" | "rank" | "len" | "labels") => {
            false
        }
        Expr::FnCall { name, args, .. } => {
            body_uses(name, sc, seen) || args.iter().any(|a| uses(a, sc, seen))
        }
        _ => crate::grad_children::children(expr)
            .is_none_or(|subs| subs.into_iter().any(|e| uses(e, sc, seen))),
    }
}

/// Whether a `u:` function's body reads a parameter. Builtins have no body.
/// An unknown `u:` name is conservatively param-using. `seen` breaks recursion:
/// a function already being analyzed contributes nothing new to the OR.
fn body_uses(name: &str, sc: &Scope, seen: &mut HashSet<String>) -> bool {
    if !name.starts_with("u:") || !seen.insert(name.to_string()) {
        return false;
    }
    match sc.env.get_fn(name) {
        Some(f) => f.body.iter().any(|e| uses(e, sc, seen)),
        None => true,
    }
}

/// Builtins whose result is a constant leaf on the tape. Index / mask builtins
/// are non-differentiable by nature (they return integer positions or `{0,1}`
/// masks) and act as stop-gradient constants (finding F5), so a top-1 router
/// mask -- `one_hot(argmax(R, 1), E)` or a `gt`/`eq`/`lt` comparison -- can be
/// computed inside the loss while the gradient flows through the surrounding
/// ops. Constant constructors `fill` / `zeros` / `ones` (finding F14) build
/// arrays from shape/value arguments, so a loss may scale by an inline mask.
/// Shape metadata (`shape` / `rank` / `len`) reads structure, not values, so
/// it is a constant of its argument's traced forward value -- which resolves a
/// user function's parameter through the traced scope.
pub(crate) fn is_constant_leaf_builtin(name: &str) -> bool {
    matches!(
        name,
        "argmax"
            | "one_hot"
            | "eq"
            | "gt"
            | "lt"
            | "argtop_k"
            | "fill"
            | "zeros"
            | "ones"
            | "shape"
            | "rank"
            | "len"
    )
}
