use crate::env::Environment;

#[derive(Clone, Debug)]
pub(crate) struct UserFn {
    pub(crate) params: Vec<String>,
    pub(crate) body: Vec<mlpl_parser::Expr>,
}

impl Environment {
    pub(crate) fn define_fn(&mut self, name: String, f: UserFn) {
        self.user_fns.insert(name, f);
    }

    pub(crate) fn get_fn(&self, name: &str) -> Option<&UserFn> {
        self.user_fns.get(name)
    }
}
