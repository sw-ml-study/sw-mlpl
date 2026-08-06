//! Include expansion: files -> ordered chunks, spliced at the
//! include site, load-once, cycles reported with the full chain.

use std::collections::HashSet;

use mlpl_parser_ast::Expr;

use crate::provider::{IncludeError, SourceId, SourceProvider};

/// A run of statements from ONE source file. Spans inside
/// `stmts` are byte offsets into that file's own text.
#[derive(Debug)]
pub struct Chunk {
    /// Which source the statements came from.
    pub source: SourceId,
    /// The statements, in source order.
    pub stmts: Vec<Expr>,
}

/// Source texts by id, for `path:line:column` rendering.
#[derive(Debug, Default)]
pub struct SourceTable {
    entries: Vec<(SourceId, String)>,
}

impl SourceTable {
    /// The stored text for `id`, if loaded.
    #[must_use]
    pub fn text(&self, id: &SourceId) -> Option<&str> {
        self.entries
            .iter()
            .find(|(i, _)| i == id)
            .map(|(_, t)| t.as_str())
    }
}

/// Expand `root` and every reachable include into ordered chunks.
///
/// # Errors
/// Any [`IncludeError`]: unresolvable/rejected paths, cycles (with
/// the complete chain), or a file that fails to lex/parse.
pub fn expand(
    root: &SourceId,
    provider: &dyn SourceProvider,
) -> Result<(Vec<Chunk>, SourceTable), IncludeError> {
    let mut st = ExpandState {
        provider,
        chunks: Vec::new(),
        table: SourceTable::default(),
        loaded: HashSet::new(),
        stack: Vec::new(),
    };
    st.load(root)?;
    Ok((st.chunks, st.table))
}

struct ExpandState<'a> {
    provider: &'a dyn SourceProvider,
    chunks: Vec<Chunk>,
    table: SourceTable,
    loaded: HashSet<SourceId>,
    stack: Vec<SourceId>,
}

impl ExpandState<'_> {
    fn load(&mut self, id: &SourceId) -> Result<(), IncludeError> {
        if self.stack.contains(id) {
            let mut chain: Vec<String> = self.stack.iter().map(|s| s.0.clone()).collect();
            chain.push(id.0.clone());
            return Err(IncludeError::Cycle { chain });
        }
        if !self.loaded.insert(id.clone()) {
            return Ok(()); // load-once: repeats are idempotent
        }
        let text = self.provider.read(id)?;
        let stmts = parse_one(id, &text)?;
        self.table.entries.push((id.clone(), text));
        self.stack.push(id.clone());
        self.splice(id, stmts)?;
        self.stack.pop();
        Ok(())
    }

    /// Emit `stmts` as chunks, recursing into includes at their
    /// exact position so definitions land in source order.
    fn splice(&mut self, id: &SourceId, stmts: Vec<Expr>) -> Result<(), IncludeError> {
        let mut run: Vec<Expr> = Vec::new();
        for stmt in stmts {
            if let Expr::Include(rel, _) = &stmt {
                if !run.is_empty() {
                    self.chunks.push(Chunk {
                        source: id.clone(),
                        stmts: std::mem::take(&mut run),
                    });
                }
                let target = self.provider.resolve(id, rel)?;
                self.load(&target)?;
            } else {
                run.push(stmt);
            }
        }
        if !run.is_empty() {
            self.chunks.push(Chunk {
                source: id.clone(),
                stmts: run,
            });
        }
        Ok(())
    }
}

fn parse_one(id: &SourceId, text: &str) -> Result<Vec<Expr>, IncludeError> {
    let parse_err = |error| IncludeError::Parse {
        source: id.0.clone(),
        error,
    };
    let tokens = mlpl_parser::lex(text).map_err(parse_err)?;
    mlpl_parser::parse(&tokens).map_err(parse_err)
}
