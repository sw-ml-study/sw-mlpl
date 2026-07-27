use mlpl_parser::Expr;

use crate::env::Environment;

#[derive(Clone, Debug)]
pub(crate) struct UserFn {
    pub(crate) params: Vec<String>,
    pub(crate) body: Vec<Expr>,
    pub(crate) doc: Option<String>,
    /// Verbatim `def ... }` text (comments intact) when the entry
    /// point supplied the program source; `:list` prefers this.
    pub(crate) source: Option<String>,
}

impl UserFn {
    pub(crate) fn new(params: Vec<String>, body: Vec<Expr>) -> Self {
        let doc = match body.first() {
            Some(Expr::StrLit(s, _)) => Some(s.clone()),
            _ => None,
        };
        Self {
            params,
            body,
            doc,
            source: None,
        }
    }

    pub(crate) fn with_source(mut self, source: Option<String>) -> Self {
        self.source = source;
        self
    }

    pub(crate) fn body_exprs(&self) -> &[Expr] {
        if self.doc.is_some() && self.body.len() > 1 {
            &self.body[1..]
        } else {
            &self.body
        }
    }
}

impl Environment {
    /// Attach (or clear) the raw program text for the CURRENT eval
    /// so `def u:` captures its span verbatim -- entry points that
    /// cannot use `eval_source_value` call this around their eval.
    pub fn set_pending_source(&mut self, src: Option<String>) {
        self.pending_source = src;
    }

    pub(crate) fn define_fn(&mut self, name: String, f: UserFn) {
        self.user_fns.insert(name, f);
    }

    pub(crate) fn get_fn(&self, name: &str) -> Option<&UserFn> {
        self.user_fns.get(name)
    }

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

    pub fn describe_fn(&self, name: &str) -> Option<String> {
        let f = self.user_fns.get(name)?;
        let mut out = format!("def {}({})", name, f.params.join(", "));
        if let Some(d) = &f.doc {
            out.push_str(&format!("\n  \"{d}\""));
        }
        for expr in f.body_exprs() {
            out.push_str(&format!("\n  {expr}"));
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
