//! Structural children of the expression forms grad's purity walk can see
//! through (`grad_purity`). Pure AST shape, no param semantics.

use mlpl_parser::Expr;

/// The direct sub-expressions of `expr`: `Some(vec![])` for a leaf literal,
/// `None` for a form the walk does not understand (the caller treats that as
/// possibly param-using). Identifiers and calls are handled by the caller.
pub(crate) fn children(expr: &Expr) -> Option<Vec<&Expr>> {
    let one = std::iter::once;
    Some(match expr {
        Expr::BinOp { lhs, rhs, .. } => vec![lhs, rhs],
        Expr::UnaryNeg { operand: e, .. } | Expr::Assign { value: e, .. } => vec![e],
        Expr::ArrayLit(es, _) => es.iter().collect(),
        Expr::Repeat { count, body, .. } => one(count.as_ref()).chain(body).collect(),
        Expr::If {
            cond,
            then_body: t,
            else_body: e,
            ..
        } => one(cond.as_ref()).chain(t).chain(e).collect(),
        Expr::IntLit(..)
        | Expr::FloatLit(..)
        | Expr::StrLit(..)
        | Expr::BuiltinRef(..)
        | Expr::TensorCtor { .. } => vec![],
        _ => return None,
    })
}
