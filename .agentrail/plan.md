# autograd-partition

Split the autograd component to clear its module ceilings so future grad ops
add cleanly. Retire the 4 module-fn FAILs: tensor_ops(8), tensor_reduce(11),
backward_shape(8) [mlpl-autograd]; grad_kernels(9) [mlpl-autograd-tape]. Both
crates are at 7 modules (ceiling). Target: a new mlpl-autograd-backward crate
+ intra-crate splits, keeping every crate <=7 modules and acyclic
(array < tape < backward < autograd). Pure refactor: NO behavior change; the
full autograd + eval grad suites stay green after every step. Measure
sw-checklist before/after; target a net FAIL reduction.

## Steps

1. extract-autograd-backward-crate -- create mlpl-autograd-backward; move
   backward.rs + backward_shape.rs (split backward_shape to <=7 fns) and the
   cross_entropy kernels (forward/backward/ce_split, from tensor_ops.rs) into
   it. backward crate depends on tape + array-ops only (acyclic); autograd's
   Tensor::backward and Tensor::cross_entropy call into it. Frees autograd
   7->5 modules; retires backward_shape + tensor_ops FAILs. grad_kernels stays
   in tape for now (backward crate imports it). Full grad suites green.
2. move-grad-kernels-to-backward -- move grad_kernels.rs from mlpl-autograd-tape
   into mlpl-autograd-backward and split it to <=7 fns per module; update
   imports. tape 7->6 modules; retires grad_kernels FAIL. Green.
3. split-tensor-reduce -- split tensor_reduce.rs (11 fns) within mlpl-autograd
   (now has room) by responsibility (reductions vs derived-node constructor vs
   the mis-placed transpose_axes -> tensor_shape). Retires tensor_reduce FAIL.
   Green.
4. partition-relay-close -- confirm all 4 FAILs retired (target 34), no new
   crate/module FAIL, docs + wiki + CHANGES updated, binaries rebuilt. --done.
