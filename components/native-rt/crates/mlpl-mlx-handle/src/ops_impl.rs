//! `DeviceOps` for MLX: every method serializes on the process
//! lock, builds LAZY graph nodes (no `.eval()` anywhere here), and
//! wraps results back into shared handles.

use mlpl_array::DenseArray;
use mlpl_tensor_handle::{AxisKind, BinKind, Dev, DeviceOps, HandleError, UnaryKind};
use mlx_rs::Array as MlxArray;

use crate::buf::MlxBuf;
use crate::support::{backend_err, step_mask};

/// The MLX backend singleton.
#[derive(Debug)]
pub(crate) struct MlxOps;

impl DeviceOps for MlxOps {
    fn upload(&self, a: &DenseArray) -> Result<Dev, HandleError> {
        let _gpu = mlpl_mlx_rt::mlx_op_lock();
        Ok(MlxBuf::wrap(mlpl_mlx_rt::dense_to_mlx(
            a.data(),
            a.shape().dims(),
        )))
    }

    fn binary(&self, op: BinKind, a: &Dev, b: &Dev) -> Result<Dev, HandleError> {
        let _gpu = mlpl_mlx_rt::mlx_op_lock();
        let (x, y) = (MlxBuf::of(a)?, MlxBuf::of(b)?);
        let out = match op {
            BinKind::Add => x.add(y),
            BinKind::Sub => x.subtract(y),
            BinKind::Mul => x.multiply(y),
            BinKind::Div => x.divide(y),
            BinKind::Matmul => x.matmul(y),
        }
        .map_err(backend_err)?;
        Ok(MlxBuf::wrap(out))
    }

    fn unary(&self, op: UnaryKind, a: &Dev) -> Result<Dev, HandleError> {
        let _gpu = mlpl_mlx_rt::mlx_op_lock();
        let x = MlxBuf::of(a)?;
        let out = match op {
            UnaryKind::Neg => x.negative(),
            UnaryKind::Exp => x.exp(),
            UnaryKind::Log => x.log(),
            UnaryKind::Tanh => mlx_rs::ops::tanh(x),
            UnaryKind::Sigmoid => mlx_rs::ops::sigmoid(x),
            UnaryKind::Relu => mlx_rs::nn::relu(x),
            UnaryKind::Gtz => step_mask(x),
            UnaryKind::Sqrt => x.sqrt(),
            UnaryKind::Transpose => x.transpose(),
        }
        .map_err(backend_err)?;
        Ok(MlxBuf::wrap(out))
    }

    fn axis_op(
        &self,
        op: AxisKind,
        a: &Dev,
        axis: Option<usize>,
        keep_dims: bool,
    ) -> Result<Dev, HandleError> {
        let _gpu = mlpl_mlx_rt::mlx_op_lock();
        let x = MlxBuf::of(a)?;
        let ax = axis.map(i32::try_from).transpose().map_err(backend_err)?;
        let out = match (op, ax) {
            (AxisKind::Softmax, Some(ax)) => mlx_rs::ops::softmax_axis(x, ax, None),
            (AxisKind::LogSoftmax, Some(ax)) => mlx_rs::nn::log_softmax(x, ax),
            (AxisKind::Softmax | AxisKind::LogSoftmax, None) => {
                return Err(HandleError::Backend("softmax requires an axis".into()));
            }
            (AxisKind::Sum, Some(ax)) => x.sum_axis(ax, keep_dims),
            (AxisKind::Sum, None) => x.sum(keep_dims),
            (AxisKind::Mean, Some(ax)) => x.mean_axis(ax, keep_dims),
            (AxisKind::Mean, None) => x.mean(keep_dims),
        }
        .map_err(backend_err)?;
        Ok(MlxBuf::wrap(out))
    }

    fn reshape(&self, a: &Dev, dims: &[usize]) -> Result<Dev, HandleError> {
        let _gpu = mlpl_mlx_rt::mlx_op_lock();
        let x = MlxBuf::of(a)?;
        let shape: Vec<i32> = dims
            .iter()
            .map(|&d| i32::try_from(d))
            .collect::<Result<_, _>>()
            .map_err(backend_err)?;
        Ok(MlxBuf::wrap(x.reshape(&shape).map_err(backend_err)?))
    }

    fn full(&self, dims: &[usize], value: f64) -> Result<Dev, HandleError> {
        let _gpu = mlpl_mlx_rt::mlx_op_lock();
        let shape: Vec<i32> = dims
            .iter()
            .map(|&d| i32::try_from(d))
            .collect::<Result<_, _>>()
            .map_err(backend_err)?;
        // The f64 -> f32 narrowing lives in dense_to_mlx -- the one
        // place that owns the backend dtype contract.
        let v = mlpl_mlx_rt::dense_to_mlx(&[value], &[]);
        let arr = mlx_rs::ops::full::<f32>(&shape, &v).map_err(backend_err)?;
        Ok(MlxBuf::wrap(arr))
    }

    fn cross_entropy(&self, logits: &Dev, targets: &[usize]) -> Result<Dev, HandleError> {
        let _gpu = mlpl_mlx_rt::mlx_op_lock();
        let x = MlxBuf::of(logits)?;
        let n = i32::try_from(targets.len()).map_err(backend_err)?;
        let idx: Vec<i32> = targets
            .iter()
            .map(|&t| i32::try_from(t))
            .collect::<Result<_, _>>()
            .map_err(backend_err)?;
        let idx = MlxArray::from_slice(&idx, &[n, 1]);
        let lse = mlx_rs::ops::logsumexp_axis(x, 1, true).map_err(backend_err)?;
        let picked = mlx_rs::ops::indexing::take_along_axis(x, &idx, 1).map_err(backend_err)?;
        let out = lse
            .subtract(&picked)
            .and_then(|d| d.mean(false))
            .map_err(backend_err)?;
        Ok(MlxBuf::wrap(out))
    }
}
