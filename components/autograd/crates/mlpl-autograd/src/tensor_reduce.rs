//! Reduction methods on [`Tensor`] (sum / mean / softmax) plus the
//! shared derived-node constructor.

use std::rc::Rc;

use mlpl_array::DenseArray;
use mlpl_array_ops_compose::prelude::*;

use crate::tensor::Tensor;
use mlpl_autograd_tape::{NodeData, NodeKind, softmax_forward};

pub(crate) fn new_tensor(t: &Tensor, value: DenseArray, kind: NodeKind) -> Tensor {
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
        let s: f64 = self.value().data().iter().sum();
        new_tensor(
            self,
            DenseArray::from_scalar(s),
            NodeKind::SumAll { parent: self.node },
        )
    }

    /// Mean over all elements.
    #[must_use]
    pub fn mean(&self) -> Self {
        let v = self.value();
        let n = v.data().len() as f64;
        let s: f64 = v.data().iter().sum();
        new_tensor(
            self,
            DenseArray::from_scalar(s / n),
            NodeKind::MeanAll { parent: self.node },
        )
    }

    /// Softmax along the last axis (rank-1 or rank-2 inputs).
    #[must_use]
    pub fn softmax(&self) -> Self {
        let v = self.value();
        let axis = v.shape().rank().saturating_sub(1);
        let y = softmax_forward(&v, axis);
        new_tensor(
            self,
            y,
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
        let v = self.value().rotate(k, axis).expect("rotate: axis in range");
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
