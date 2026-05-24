//! Apply trained BPE merges to a byte string, and decode
//! token ids back to bytes.

use crate::train::apply_merge;

/// Apply a trained merge list to `bytes`, returning the
/// compressed token sequence. Walks merges in training order;
/// each merge is applied with the same greedy left-to-right
/// routine used during training.
pub fn apply_trained(bytes: &[u8], merges: &[(u32, u32)]) -> Vec<u32> {
    let mut tokens: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
    for (i, &pair) in merges.iter().enumerate() {
        let new_id = 256 + u32::try_from(i).unwrap_or(u32::MAX - 256);
        tokens = apply_merge(&tokens, pair, new_id);
    }
    tokens
}

/// Recursively expand a single BPE token id back into its
/// byte sequence. Byte ids (< 256) decode to themselves;
/// merged ids (>= 256) recursively expand their (left, right)
/// pair.
pub fn decode_token(id: u32, merges: &[(u32, u32)], out: &mut Vec<u8>) {
    if id < 256 {
        out.push(id as u8);
        return;
    }
    let merge_idx = (id - 256) as usize;
    if merge_idx >= merges.len() {
        // Unknown merged id -- skip; defensive. Valid trained
        // outputs never produce ids outside this range.
        return;
    }
    let (l, r) = merges[merge_idx];
    decode_token(l, merges, out);
    decode_token(r, merges, out);
}
