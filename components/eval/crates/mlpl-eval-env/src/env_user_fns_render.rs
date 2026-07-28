//! `u:` function LISTING/RENDER accessors (`:fns`, `:describe`,
//! `:list`), split from `env_user_fns` to stay inside the module
//! function-count budget. Definition/lookup stays there.

use std::fmt::Write as _;

use crate::env::Environment;

impl Environment {
    #[must_use]
    pub fn user_fn_signatures(&self) -> Vec<String> {
        let mut out: Vec<_> = self
            .user_fns
            .iter()
            .map(|(name, f)| {
                let sig = format!("{}({})", name, f.params.join(", "));
                match &f.doc {
                    Some(d) => format!("{sig}  -- {d}"),
                    None => sig,
                }
            })
            .collect();
        out.sort();
        out
    }

    #[must_use]
    pub fn describe_fn(&self, name: &str) -> Option<String> {
        let f = self.user_fns.get(name)?;
        let mut out = format!("def {}({})", name, f.params.join(", "));
        if let Some(d) = &f.doc {
            let _ = write!(out, "\n  \"{d}\"");
        }
        for expr in f.body_exprs() {
            let _ = write!(out, "\n  {expr}");
        }
        Some(out)
    }

    /// The full `def` source for `:list <fn>`, re-indented so control flow
    /// (`if`/`else`/`while`/`for`) reads instead of running off one flat
    /// line. `None` if no such user function is defined.
    pub fn list_fn(&self, name: &str) -> Option<String> {
        let f = self.user_fns.get(name)?;
        if let Some(src) = &f.source {
            return Some(src.clone());
        }
        let body: Vec<String> = f.body.iter().map(ToString::to_string).collect();
        let flat = format!(
            "def {}({}) {{ {} }}",
            name,
            f.params.join(", "),
            body.join("; ")
        );
        Some(mlpl_eval_core::indent_source(&flat))
    }
}
