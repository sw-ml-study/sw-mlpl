//! Block-level (paragraph / list) parsing for the
//! markdown-ish path-body renderer.

use super::block_acc::BlockAcc;

#[derive(Debug)]
pub(crate) enum Block {
    Paragraph(String),
    Unordered(Vec<String>),
    Ordered(Vec<String>),
}

pub(crate) fn parse_blocks(body: &str) -> Vec<Block> {
    let mut acc = BlockAcc::default();
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            acc.flush_all();
        } else if let Some(rest) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            acc.push_ul(rest);
        } else if let Some(rest) = strip_ol_marker(trimmed) {
            acc.push_ol(rest);
        } else {
            acc.push_para_line(trimmed);
        }
    }
    acc.flush_all();
    acc.blocks
}

fn strip_ol_marker(s: &str) -> Option<&str> {
    let mut chars = s.char_indices();
    let mut last_digit_end = 0;
    let mut saw_digit = false;
    for (i, c) in chars.by_ref() {
        if c.is_ascii_digit() {
            saw_digit = true;
            last_digit_end = i + 1;
        } else {
            break;
        }
    }
    if !saw_digit {
        return None;
    }
    let after_digits = &s[last_digit_end..];
    after_digits.strip_prefix(". ")
}

#[cfg(test)]
mod tests {
    use super::{Block, parse_blocks};

    #[test]
    fn empty_body_is_no_blocks() {
        assert!(parse_blocks("").is_empty());
        assert!(parse_blocks("   \n\n  ").is_empty());
    }

    #[test]
    fn single_paragraph_is_one_block() {
        let blocks = parse_blocks("Hello world, this is one paragraph.");
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], Block::Paragraph(s) if s.contains("Hello world")));
    }

    #[test]
    fn blank_line_splits_paragraphs() {
        let blocks = parse_blocks("first\n\nsecond\n\nthird");
        assert_eq!(blocks.len(), 3);
        for b in &blocks {
            assert!(matches!(b, Block::Paragraph(_)));
        }
    }

    #[test]
    fn bulleted_lines_collapse_to_one_list() {
        let blocks = parse_blocks("intro\n\n- a\n- b\n- c\n\nouttro");
        assert_eq!(blocks.len(), 3);
        match &blocks[1] {
            Block::Unordered(items) => assert_eq!(items.len(), 3),
            other => panic!("expected Unordered, got {other:?}"),
        }
    }

    #[test]
    fn numbered_lines_become_ordered() {
        let blocks = parse_blocks("steps:\n\n1. first\n2. second\n3. third");
        assert_eq!(blocks.len(), 2);
        match &blocks[1] {
            Block::Ordered(items) => assert_eq!(items.len(), 3),
            other => panic!("expected Ordered, got {other:?}"),
        }
    }

    #[test]
    fn paragraph_continues_across_single_newlines() {
        let blocks = parse_blocks("line one\nline two\nline three");
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            Block::Paragraph(text) => assert!(text.contains("line three")),
            other => panic!("expected Paragraph, got {other:?}"),
        }
    }
}
