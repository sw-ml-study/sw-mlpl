//! Argument coercion for freeze / unfreeze: validate arity
//! and produce a `ModelSpec`. The bare-ident path goes
//! directly through `HasModels::get_model`; any other
//! expression flows through the caller-supplied `resolve`
//! closure so this crate never imports the wider eval loop.

use mlpl_env_traits::HasModels;
use mlpl_eval_core::model::ModelSpec;
use mlpl_parser::Expr;

use crate::error::FreezeError;

pub(crate) fn resolve_model<E, F, Err>(
    args: &[Expr],
    env: &mut E,
    resolve: F,
    func: &str,
) -> Result<ModelSpec, Err>
where
    E: HasModels,
    F: FnOnce(&Expr, &mut E) -> Result<ModelSpec, Err>,
    Err: From<FreezeError>,
{
    let [arg] = args else {
        return Err(bad_arity(func, args.len()).into());
    };
    if let Expr::Ident(name, _) = arg {
        return env
            .get_model(name)
            .cloned()
            .ok_or_else(|| not_a_model(func, name).into());
    }
    resolve(arg, env)
}

fn bad_arity(func: &str, got: usize) -> FreezeError {
    FreezeError::BadArity {
        func: func.into(),
        expected: 1,
        got,
    }
}

fn not_a_model(func: &str, name: &str) -> FreezeError {
    FreezeError::NotAModel {
        func: func.into(),
        name: name.into(),
    }
}
