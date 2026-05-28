use mlpl_array::DenseArray;
use mlpl_runtime_core::error::RuntimeError;

pub const NAMES: &[&str] = &[
    "pi",
    "e",
    "exp",
    "log",
    "sqrt",
    "abs",
    "sin",
    "cos",
    "sigmoid",
    "tanh_fn",
    "relu",
    "floor",
    "ceil",
    "round",
    "pow",
    "mod",
    "gt",
    "lt",
    "eq",
    "mean",
    "zeros",
    "ones",
    "fill",
    "cosine_schedule",
    "linear_warmup",
    "concat",
    "last_row",
];

pub fn try_call(name: &str, args: Vec<DenseArray>) -> Option<Result<DenseArray, RuntimeError>> {
    use crate::array_ops as ao;
    use crate::elementwise as ew;
    match name {
        "pi" => Some(ew::zero_arg(name, args, std::f64::consts::PI)),
        "e" => Some(ew::zero_arg(name, args, std::f64::consts::E)),
        "exp" => Some(ew::unary(name, args, f64::exp)),
        "log" => Some(ew::unary(name, args, f64::ln)),
        "sqrt" => Some(ew::unary(name, args, f64::sqrt)),
        "abs" => Some(ew::unary(name, args, f64::abs)),
        "sin" => Some(ew::unary(name, args, f64::sin)),
        "cos" => Some(ew::unary(name, args, f64::cos)),
        "sigmoid" => Some(ew::unary(name, args, |x| 1.0 / (1.0 + (-x).exp()))),
        "tanh_fn" => Some(ew::unary(name, args, f64::tanh)),
        "relu" => Some(ew::unary(name, args, |x| x.max(0.0))),
        "floor" => Some(ew::unary(name, args, f64::floor)),
        "ceil" => Some(ew::unary(name, args, f64::ceil)),
        "round" => Some(ew::unary(name, args, f64::round)),
        "pow" => Some(ew::binary_pow(name, args)),
        "mod" => Some(ew::binary_cmp(name, args, |a, b| a % b)),
        "gt" => Some(ew::binary_cmp(
            name,
            args,
            |a, b| {
                if a > b { 1.0 } else { 0.0 }
            },
        )),
        "lt" => Some(ew::binary_cmp(
            name,
            args,
            |a, b| {
                if a < b { 1.0 } else { 0.0 }
            },
        )),
        "eq" => Some(ew::binary_cmp(name, args, |a, b| {
            if (a - b).abs() < f64::EPSILON {
                1.0
            } else {
                0.0
            }
        })),
        "mean" => Some(builtin_mean(name, args)),
        "zeros" | "ones" | "fill" => Some(ao::constructor(name, args)),
        "cosine_schedule" | "linear_warmup" => Some(ao::schedule(name, args)),
        "concat" if args.len() != 3 => Some(ao::array_util(name, args)),
        "last_row" => Some(ao::array_util(name, args)),
        _ => None,
    }
}

fn builtin_mean(name: &str, args: Vec<DenseArray>) -> Result<DenseArray, RuntimeError> {
    if args.len() != 1 {
        return Err(RuntimeError::ArityMismatch {
            func: name.into(),
            expected: 1,
            got: args.len(),
        });
    }
    let data = args[0].data();
    let sum: f64 = data.iter().sum();
    let count = data.len().max(1) as f64;
    Ok(DenseArray::from_scalar(sum / count))
}
