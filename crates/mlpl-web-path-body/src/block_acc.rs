//! Accumulator struct + impl for the markdown-ish block
//! parser. Split into its own file so blocks.rs stays under
//! the sw-checklist per-module function-count budget.

use super::blocks::Block;

#[derive(Default)]
pub(super) struct BlockAcc<'a> {
    pub(super) blocks: Vec<Block>,
    para_lines: Vec<&'a str>,
    ul_items: Vec<String>,
    ol_items: Vec<String>,
}

impl<'a> BlockAcc<'a> {
    pub(super) fn push_para_line(&mut self, line: &'a str) {
        self.flush_ul();
        self.flush_ol();
        self.para_lines.push(line);
    }
    pub(super) fn push_ul(&mut self, text: &str) {
        self.flush_para();
        self.flush_ol();
        self.ul_items.push(text.to_string());
    }
    pub(super) fn push_ol(&mut self, text: &str) {
        self.flush_para();
        self.flush_ul();
        self.ol_items.push(text.to_string());
    }
    pub(super) fn flush_all(&mut self) {
        self.flush_para();
        self.flush_ul();
        self.flush_ol();
    }
    fn flush_para(&mut self) {
        if !self.para_lines.is_empty() {
            self.blocks
                .push(Block::Paragraph(self.para_lines.join(" ")));
            self.para_lines.clear();
        }
    }
    fn flush_ul(&mut self) {
        if !self.ul_items.is_empty() {
            self.blocks
                .push(Block::Unordered(std::mem::take(&mut self.ul_items)));
        }
    }
    fn flush_ol(&mut self) {
        if !self.ol_items.is_empty() {
            self.blocks
                .push(Block::Ordered(std::mem::take(&mut self.ol_items)));
        }
    }
}
