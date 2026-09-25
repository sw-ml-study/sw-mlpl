//! Native row gather on the tape: `gather_rows(table, indices)` copies the
//! addressed rows forward and scatter-ADDS the upstream rows back into
//! only the touched table rows -- O(n * d) both ways, replacing the dense
//! one-hot selection matmul (O(n * rows * d) time and n * rows memory),
//! which cost ~30 s per step at embedding-scale `n` and vocabularies.
//! On a resident (device) tape the gather runs on the host (a counted
//! CPU fallback); there is no device kernel for it.

use mlpl_array::{DenseArray, Shape};
use mlpl_autograd_tape::NodeKind;
use mlpl_tensor_handle::TensorHandle;

use crate::tensor::Tensor;
use crate::tensor_reduce::new_tensor;

impl Tensor {
    /// Gather rows `indices` (each `< rows`) of this rank-2 `[rows, d]`
    /// tensor into `[indices.len(), d]`. The caller validates the indices
    /// and the rank.
    #[must_use]
    pub fn gather_rows(&self, indices: Vec<usize>) -> Self {
        if self.tape.resident.get() {
            mlpl_tensor_handle::bump(mlpl_tensor_handle::SeamEvent::CpuFallback);
        }
        let table = self.value();
        let (rows, d) = (table.shape().dims()[0], table.shape().dims()[1]);
        let data: Vec<f64> = indices
            .iter()
            .flat_map(|&r| table.data()[r * d..(r + 1) * d].iter().copied())
            .collect();
        let out = DenseArray::new(Shape::new(vec![indices.len(), d]), data)
            .expect("gather_rows: n * d elements");
        let kind = NodeKind::GatherRows {
            parent: self.node,
            indices,
            rows,
        };
        new_tensor(self, TensorHandle::Cpu(out), kind)
    }
}
