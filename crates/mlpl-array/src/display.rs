//! Display formatting for DenseArray.

use crate::dense::DenseArray;
use std::fmt;

impl fmt::Display for DenseArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.elem_count() == 0 {
            return write!(f, "[]");
        }
        match self.rank() {
            0 => write!(f, "{}", self.data()[0]),
            1 => write_row(f, self.data()),
            2 => {
                let cols = self.shape().dims()[1];
                for (i, row) in self.data().chunks(cols).enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write_row(f, row)?;
                }
                Ok(())
            }
            _ => write!(f, "{:?}", self.data()),
        }
    }
}

/// Format a single row of f64 values, space-separated.
fn write_row(f: &mut fmt::Formatter<'_>, values: &[f64]) -> fmt::Result {
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            write!(f, " ")?;
        }
        write!(f, "{v}")?;
    }
    Ok(())
}
