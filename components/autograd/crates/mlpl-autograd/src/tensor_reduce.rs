//! Reduction methods on [`Tensor`] (sum / mean / reduce_sum) plus the shared
//! derived-node constructors. The axis/shape-transforming ops (softmax,
//! transpose_axes, rotate, windows) live in `tensor_transform`.

use std::rc::Rc;

use mlpl_array::DenseArray;
use mlpl_array_ops_reduce::prelude::*;

use crate::tensor::Tensor;
use mlpl_autograd_tape::{NodeData, NodeKind, ResidentReq, Tape, try_resident};
use mlpl_tensor_handle::{AxisKind, TensorHandle};

/// Trainable leaf whose forward value is an EXISTING handle --
/// resident optimizer seeding reuses last step's device weight
/// without re-uploading (saga E4 step 006).
#[must_use]
pub fn param_from_handle(tape: Rc<Tape>, value: TensorHandle) -> Tensor {
    let node = tape.push(NodeData {
        value,
        grad: None,
        kind: NodeKind::Leaf,
        requires_grad: true,
    });
    Tensor { node, tape }
}

/// The accumulated gradient as a HANDLE (no materialization) --
/// the resident optimizer consumes it device-side.
#[must_use]
pub fn grad_handle_of(t: &Tensor) -> Option<TensorHandle> {
    t.tape.nodes()[t.node.0].grad.clone()
}

pub(crate) fn new_tensor(t: &Tensor, value: TensorHandle, kind: NodeKind) -> Tensor {
    let node = t.tape.push(NodeData {
        value,
        grad: None,
        kind,
        requires_grad: false,
    });
    Tensor {
        node,
        tape: Rc::clone(&t.tape),
    }
}

impl Tensor {
    /// Sum all elements into a scalar.
    #[must_use]
    pub fn sum(&self) -> Self {
        let value = try_resident(
            &self.tape,
            ResidentReq::Axis(self.node, AxisKind::Sum, None, false),
        )
        .unwrap_or_else(|| {
            TensorHandle::Cpu(DenseArray::from_scalar(self.value().data().iter().sum()))
        });
        new_tensor(self, value, NodeKind::SumAll { parent: self.node })
    }

    /// Mean over all elements.
    #[must_use]
    pub fn mean(&self) -> Self {
        let value = try_resident(
            &self.tape,
            ResidentReq::Axis(self.node, AxisKind::Mean, None, false),
        )
        .unwrap_or_else(|| {
            let v = self.value();
            let s: f64 = v.data().iter().sum();
            TensorHandle::Cpu(DenseArray::from_scalar(s / v.data().len() as f64))
        });
        new_tensor(self, value, NodeKind::MeanAll { parent: self.node })
    }

    /// Sum over one or more `axes` (the differentiable core of
    /// `reduce(:add, x, axes)`). Backward broadcasts the gradient back
    /// over the reduced axes. Axes collapse high-index first so earlier
    /// removals do not shift the rest.
    pub fn reduce_sum(&self, axes: &[usize]) -> Self {
        let v_orig = self.value();
        let orig_shape = v_orig.shape().clone();
        if self.tape.resident.get() {
            mlpl_tensor_handle::bump(mlpl_tensor_handle::SeamEvent::CpuFallback);
        }
        let mut sorted = axes.to_vec();
        sorted.sort_unstable();
        let reduced = sorted.iter().rev().fold(v_orig, |acc, &ax| {
            acc.reduce_axis(ax, 0.0, |a, b| a + b)
                .expect("reduce_sum: axis in range")
        });
        new_tensor(
            self,
            TensorHandle::Cpu(reduced),
            NodeKind::ReduceSum {
                parent: self.node,
                orig_shape,
                axes: sorted,
            },
        )
    }
}
