//! Descriptor validation, separated from the registration effect.
//! Pure checks over the raw C fields: bounded UTF-8 text, and one
//! function entry's name / reserved / invoke. Dedup is the
//! caller's job (it owns the seen-set across the table).

use std::collections::BTreeSet;

use crate::marshal::read_abi_slice;
use crate::model::{AbiSlice, FunctionDescriptorV1, InvokeFnV1};

/// Largest UTF-8 text field (name / version / function name).
const MAX_TEXT_BYTES: usize = 16 * 1024;

/// Read a required, non-empty, bounded UTF-8 text field.
pub(crate) fn read_required_text(label: &str, slice: AbiSlice) -> Result<String, String> {
    let bytes = read_abi_slice(label, slice)?;
    if bytes.is_empty() {
        return Err(format!("{label} is empty"));
    }
    if bytes.len() > MAX_TEXT_BYTES {
        return Err(format!("{label} exceeds {MAX_TEXT_BYTES} bytes"));
    }
    String::from_utf8(bytes).map_err(|_| format!("{label} is not valid UTF-8"))
}

/// Validate one function entry into its `(name, invoke)`,
/// rejecting a reserved-field, duplicate name, or null invoke.
/// `seen` accumulates names across the table for the dup check.
pub(crate) fn convert_function(
    fd: &FunctionDescriptorV1,
    seen: &mut BTreeSet<String>,
) -> Result<(String, InvokeFnV1), String> {
    let name = read_required_text("function name", fd.name)?;
    if fd.reserved != 0 {
        return Err(format!("function {name}: reserved field must be zero"));
    }
    if !seen.insert(name.clone()) {
        return Err(format!("duplicate function name {name}"));
    }
    let invoke = fd
        .invoke
        .ok_or_else(|| format!("function {name} has a null invoke pointer"))?;
    Ok((name, invoke))
}
