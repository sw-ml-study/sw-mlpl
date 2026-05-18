//! Reduction, shape, and matmul methods on [`Tensor`].

use std::rc::Rc;

use mlpl_array::{DenseArray, Shape};

use crate::ops::softmax_forward;
use crate::tape::{NodeData, NodeKind};
use crate::tensor::Tensor;

fn new_tensor(t: &Tensor, value: DenseArray, kind: NodeKind) -> Tensor {
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

    /// Transpose: reverse axes.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let v = self.value().transpose();
        new_tensor(self, v, NodeKind::Transpose { parent: self.node })
    }

    /// Reshape to `new_shape` (must preserve element count).
    #[must_use]
    pub fn reshape(&self, new_shape: Shape) -> Self {
        let v_orig = self.value();
        let orig_shape = v_orig.shape().clone();
        let v = v_orig.reshape(new_shape).expect("compatible reshape");
        new_tensor(
            self,
            v,
            NodeKind::Reshape {
                parent: self.node,
                orig_shape,
            },
        )
    }

    /// Matrix multiplication `self @ other`.
    #[must_use]
    pub fn matmul(&self, other: &Self) -> Self {
        let v = self
            .value()
            .matmul(&other.value())
            .expect("compatible matmul shapes");
        new_tensor(
            self,
            v,
            NodeKind::MatMul {
                left: self.node,
                right: other.node,
            },
        )
    }

    /// Saga 29 step 005: ViT patch embedding.
    /// `self` must be rank-4 `[B, C, H, W]` and `p` must
    /// divide both `H` and `W`. Returns `[B, N, P*P*C]`.
    #[must_use]
    pub fn patchify(&self, p: usize) -> Self {
        let v_orig = self.value();
        let orig_shape = v_orig.shape().clone();
        let v = v_orig.patchify(p).expect("patchify compatible shape");
        new_tensor(
            self,
            v,
            NodeKind::Patchify {
                parent: self.node,
                orig_shape,
                patch_size: p,
            },
        )
    }

    /// Saga 29 step 005: concat `self` with `other` along
    /// `axis` (0 or 1 supported in this initial release).
    #[must_use]
    pub fn concat(&self, other: &Self, axis: usize) -> Self {
        let a = self.value();
        let b = other.value();
        let left_size = a.shape().dims()[axis];
        let v = a.concat(&b, axis).expect("concat compatible shapes");
        new_tensor(
            self,
            v,
            NodeKind::Concat {
                left: self.node,
                right: other.node,
                axis,
                left_size,
            },
        )
    }
}
