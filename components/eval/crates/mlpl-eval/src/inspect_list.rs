//! `:list <fn-name>` -- render the indented `def` source of one user
//! function (see `Environment::list_fn`). Split out of `inspect.rs` so the
//! command dispatcher stays within its per-function LOC budget.

use crate::env::Environment;
use crate::env_api::*;

/// `:describe <name> [<name> ...]` -- one describe body per name,
/// blank-line separated. Lives beside the other small command
/// handlers so the dispatcher stays within its budgets.
pub(crate) fn describe_names(env: &Environment, args: &[&str]) -> String {
    if args.is_empty() {
        return "usage: :describe <name> [<name> ...]".into();
    }
    args.iter()
        .map(|n| crate::inspect_describe::format_describe(env, n))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// `:untag <name>` -- clear a binding's auto-attached tag. Lives
/// beside `:list` so the command dispatcher stays within its
/// module function budget.
pub(crate) fn handle_untag(env: &mut Environment, arg: Option<&str>) -> String {
    let Some(name) = arg else {
        return "usage: :untag <name>".into();
    };
    if env.get_tag(name).is_some() {
        env.clear_tag(name);
        format!("untagged {name}")
    } else {
        format!("{name} had no tag")
    }
}

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
