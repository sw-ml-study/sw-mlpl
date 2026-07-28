use mlpl_parser::Expr;

use crate::env::Environment;

#[derive(Clone, Debug)]
pub struct UserFn {
    pub params: Vec<String>,
    pub body: Vec<Expr>,
    pub doc: Option<String>,
    /// Verbatim `def ... }` text (comments intact) when the entry
    /// point supplied the program source; `:list` prefers this.
    pub source: Option<String>,
}

impl UserFn {
    #[must_use]
    pub fn new(params: Vec<String>, body: Vec<Expr>) -> Self {
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

    #[must_use]
    pub fn with_source(mut self, source: Option<String>) -> Self {
        self.source = source;
        self
    }

    #[must_use]
    pub fn body_exprs(&self) -> &[Expr] {
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

    pub fn define_fn(&mut self, name: String, f: UserFn) {
        self.user_fns.insert(name, f);
    }

    #[must_use]
    pub fn get_fn(&self, name: &str) -> Option<&UserFn> {
        self.user_fns.get(name)
    }
}
