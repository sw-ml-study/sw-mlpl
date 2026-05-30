# Extract image_* from mlpl-eval (saga 71)

First Phase-1 god-crate-decomposition saga. Move the 2 image_io
files into components/eval-image/. Image code only uses
EvalError::Unsupported(String); avoid cycle by defining
mlpl-eval-image's own ImageError type and `impl From<ImageError>
for EvalError` in mlpl-eval.
