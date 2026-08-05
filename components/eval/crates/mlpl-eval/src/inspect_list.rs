//! `:list <fn-name>` -- render the indented `def` source of one user
//! function (see `Environment::list_fn`). Split out of `inspect.rs` so the
//! command dispatcher stays within its per-function LOC budget.

use crate::env::Environment;
use crate::env_api::*;

/// Render `:list <arg>`: the formatted function source, a not-found note,
/// or the usage line when no name was given.
pub(crate) fn list_or_usage(env: &Environment, arg: Option<&str>) -> String {
    match arg {
        None => "usage: :list <fn-name>".into(),
        Some(name) => env.list_fn(name).unwrap_or_else(|| {
            let bare = name.strip_prefix(':').unwrap_or(name);
            if mlpl_eval_core::inspect_groups::documented_builtin_names().any(|n| n == bare) {
                format!("`{bare}` is a builtin, not a `u:` function -- try `:describe {bare}`")
            } else {
                format!("no user function named '{name}'")
            }
        }),
    }
}
