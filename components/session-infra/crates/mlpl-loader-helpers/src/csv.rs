//! CSV parser: tabular `String` -> rank-2 `DenseArray`.
//! Comma-delimited only. Auto-detects + skips a non-numeric
//! header row; ragged rows or non-numeric data rows surface
//! as structured errors.

use mlpl_array::{DenseArray, Shape};

use crate::error::LoaderHelperError;

pub fn parse_csv(text: &str, path: &str) -> Result<DenseArray, LoaderHelperError> {
    let rows = collect_data_rows(text, path)?;
    let cols = rows[0].len();
    let data = flatten_rows(&rows, cols, path)?;
    Ok(DenseArray::new(Shape::new(vec![rows.len(), cols]), data)?)
}

/// Collect non-empty, comma-split rows; auto-detect + drop a
/// non-numeric header row.
fn collect_data_rows(text: &str, path: &str) -> Result<Vec<Vec<String>>, LoaderHelperError> {
    let mut rows: Vec<Vec<String>> = text
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| line.split(',').map(|c| c.trim().to_string()).collect())
        .collect();
    if rows.is_empty() {
        return Err(LoaderHelperError::NoDataRows { path: path.into() });
    }
    let first_is_header = rows[0].iter().any(|cell| cell.parse::<f64>().is_err());
    if first_is_header {
        rows.remove(0);
        if rows.is_empty() {
            return Err(LoaderHelperError::HeaderOnly { path: path.into() });
        }
    }
    Ok(rows)
}

/// Flatten rows into a row-major f64 buffer, validating
/// column count + numeric content as we go.
fn flatten_rows(
    rows: &[Vec<String>],
    cols: usize,
    path: &str,
) -> Result<Vec<f64>, LoaderHelperError> {
    let mut data = Vec::with_capacity(rows.len() * cols);
    for (row_idx, row) in rows.iter().enumerate() {
        if row.len() != cols {
            return Err(LoaderHelperError::RaggedRow {
                path: path.into(),
                row_idx,
                got_cols: row.len(),
                expected_cols: cols,
            });
        }
        push_row(&mut data, row, row_idx, path)?;
    }
    Ok(data)
}

/// Parse one row's cells as f64s into `data`.
fn push_row(
    data: &mut Vec<f64>,
    row: &[String],
    row_idx: usize,
    path: &str,
) -> Result<(), LoaderHelperError> {
    for cell in row {
        let v: f64 = cell
            .parse()
            .map_err(|_| LoaderHelperError::NonNumericCell {
                path: path.into(),
                row_idx,
                cell: cell.clone(),
            })?;
        data.push(v);
    }
    Ok(())
}
