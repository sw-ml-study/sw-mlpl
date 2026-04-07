#[derive(Clone, PartialEq)]
pub struct HistoryEntry {
    pub input: String,
    pub output: String,
    pub is_error: bool,
}

#[derive(Clone, Copy, PartialEq)]
pub enum DocTab {
    LangReference,
    Usage,
}
