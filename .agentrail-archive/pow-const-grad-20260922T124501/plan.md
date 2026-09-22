# pow-const-grad

Make pow(x, k) differentiable inside grad for ANY constant exponent k
(fractional, negative, large), not just small positive integers. The
autograd-partition cleared the module ceilings, so a dedicated PowConst tape
node can now be added cleanly. d/dx x^k = k * x^(k-1).

## Steps

1. powconst-node -- add NodeKind::PowConst { parent, exp: f64 } to
   mlpl-autograd-tape; add Tensor::pow_const(k) forward (host x.powf(k), no
   device kernel) in mlpl-autograd; add the backward (prop_pow_const in
   mlpl-autograd-backward, elementwise upstream * k * x^(k-1)) wired via a
   propagate() arm; rewire grad_calls_pow::call_pow to build PowConst for any
   constant exponent (keep the differentiable-exponent guard). Drop the
   integer-only restriction and repeated-mul. TDD: gradcheck pow(x,0.5),
   pow(x,-1), pow(x,2.5), pow(x,3) vs finite differences; x^2 still exact;
   eager pow unchanged. Rebuild repl/build release+debug + serve.
2. relay-close -- update docs/sw-mlpl-findings.md (RS1-pow now general) +
   docs/future-sagas-queue.md; refresh CHANGES; wiki note if needed. --done.
