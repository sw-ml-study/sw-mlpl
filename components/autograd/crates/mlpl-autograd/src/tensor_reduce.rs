//! Reduction methods on [`Tensor`] (sum / mean / softmax) plus the
//! shared derived-node constructor.

use std::rc::Rc;

use mlpl_array::DenseArray;

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
