use mlpl_parser::Expr;

use crate::env::Environment;

#[derive(Clone, Debug)]
pub(crate) struct UserFn {
    pub(crate) params: Vec<String>,
    pub(crate) body: Vec<Expr>,
    pub(crate) doc: Option<String>,
}

impl UserFn {
    pub(crate) fn new(params: Vec<String>, body: Vec<Expr>) -> Self {
        let doc = match body.first() {
            Some(Expr::StrLit(s, _)) => Some(s.clone()),
            _ => None,
        };
        Self { params, body, doc }
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
}
