//! `:list <fn-name>` -- render the indented `def` source of one user
//! function (see `Environment::list_fn`). Split out of `inspect.rs` so the
//! command dispatcher stays within its per-function LOC budget.

use crate::env::Environment;

/// Render `:list <arg>`: the formatted function source, a not-found note,
/// or the usage line when no name was given.
pub(crate) fn list_or_usage(env: &Environment, arg: Option<&str>) -> String {
    match arg {
        None => "usage: :list <fn-name>".into(),
        Some(name) => env
            .list_fn(name)
            .unwrap_or_else(|| format!("no user function named '{name}'")),
    }
}
