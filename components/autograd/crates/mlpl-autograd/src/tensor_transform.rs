//! Axis- and shape-transforming tape ops on [`Tensor`]: softmax (and its
//! axis form), general-permutation transpose, cyclic rotate, and the
//! sliding-window gather. Split from `tensor_reduce` (autograd-partition) --
//! the pure reductions (sum/mean/reduce_sum) and the derived-node constructor
//! stay there; these build a NodeKind for an axis/shape transform.

use mlpl_array_ops_compose::prelude::*;
use mlpl_array_ops_shape::prelude::*;

use crate::tensor::Tensor;
use crate::tensor_reduce::new_tensor;
use mlpl_autograd_tape::{NodeKind, ResidentReq, softmax_forward, try_resident};
use mlpl_tensor_handle::{AxisKind, TensorHandle};

impl Tensor {
    /// Softmax along the last axis (rank-1 or rank-2 inputs).
    #[must_use]
    pub fn softmax(&self) -> Self {
        let axis = self.tape.nodes()[self.node.0]
            .value
            .dims()
            .len()
            .saturating_sub(1);
        self.softmax_axis(axis)
    }

    /// Softmax along an explicit `axis` (any rank; RS2). `softmax()` is the
    /// last-axis special case.
    #[must_use]
    pub fn softmax_axis(&self, axis: usize) -> Self {
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

    /// General axis-permutation transpose (RS3): output axis `i` is input
    /// axis `perm[i]`. Backward permutes the gradient by the inverse of
    /// `perm`. Caller (the grad dispatch) validates `perm`. The reverse-axes
    /// `transpose` lives in `tensor_shape`.
    #[must_use]
    pub fn transpose_axes(&self, perm: Vec<usize>) -> Self {
        if self.tape.resident.get() {
            mlpl_tensor_handle::bump(mlpl_tensor_handle::SeamEvent::CpuFallback);
        }
        let v = TensorHandle::Cpu(
            self.value()
                .transpose_axes(&perm)
                .expect("caller validated the axis permutation"),
        );
        new_tensor(
            self,
            v,
            NodeKind::Transpose {
                parent: self.node,
                perm: Some(perm),
            },
        )
    }

    /// Cyclic rotate along `axis` (positive `k` = element `k` to the front).
    /// Pure permutation; backward is `rotate(-k)`.
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

    /// Overlapping sliding-window gather over the trailing axes (CNN
    /// Phase 5). Records the parent shape + window params so the backward
    /// can scatter-add the gradient.
    pub fn windows(&self, sizes: &[usize], strides: &[usize]) -> Self {
        let v_orig = self.value();
        let orig_shape = v_orig.shape().clone();
        if self.tape.resident.get() {
            mlpl_tensor_handle::bump(mlpl_tensor_handle::SeamEvent::CpuFallback);
        }
        let v = TensorHandle::Cpu(
            v_orig
                .windows(sizes, strides)
                .expect("windows: valid params"),
        );
        new_tensor(
            self,
            v,
            NodeKind::Windows {
                parent: self.node,
                orig_shape,
                sizes: sizes.to_vec(),
                strides: strides.to_vec(),
            },
        )
    }
}
