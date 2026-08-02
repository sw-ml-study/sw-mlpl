//! Reduction methods on [`Tensor`] (sum / mean / softmax) plus the
//! shared derived-node constructor.

use std::rc::Rc;

use mlpl_array::DenseArray;
use mlpl_array_ops_compose::prelude::*;

use crate::tensor::Tensor;
use mlpl_autograd_tape::{NodeData, NodeKind, ResidentReq, Tape, softmax_forward, try_resident};
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

    /// Softmax along the last axis (rank-1 or rank-2 inputs).
    #[must_use]
    pub fn softmax(&self) -> Self {
        let axis = self.tape.nodes()[self.node.0]
            .value
            .dims()
            .len()
            .saturating_sub(1);
        let value = try_resident(
            &self.tape,
            ResidentReq::Axis(self.node, AxisKind::Softmax, Some(axis), false),
        )
        .unwrap_or_else(|| {
            if self.tape.resident.get() {
                mlpl_tensor_handle::bump(mlpl_tensor_handle::SeamEvent::CpuFallback);
            }
            TensorHandle::Cpu(softmax_forward(&self.value(), axis))
        });
        new_tensor(
            self,
            value,
            NodeKind::Softmax {
                parent: self.node,
                axis,
            },
        )
    }
}

impl Tensor {
    /// Cyclic rotate along `axis` (positive `k` = element `k` to
    /// the front). Pure permutation; backward is `rotate(-k)`.
    /// Lives here rather than in `tensor_shape.rs` because that
    /// module sits at the sw-checklist function-count cap; a
    /// future rebalance can regroup the composition methods.
    #[must_use]
    pub fn rotate(&self, k: i64, axis: usize) -> Self {
        if self.tape.resident.get() {
            mlpl_tensor_handle::bump(mlpl_tensor_handle::SeamEvent::CpuFallback);
        }
        let v = TensorHandle::Cpu(self.value().rotate(k, axis).expect("rotate: axis in range"));
        new_tensor(
            self,
            v,
            NodeKind::Rotate {
                parent: self.node,
                k,
                axis,
            },
        )
    }
}
