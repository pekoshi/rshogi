//! CUDA Context + Stream + cuBLAS 初期化

use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaStream};

/// PolicyValueNetCuda が必要とする実行コンテキスト
pub struct PolicyValueCudaContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
}

impl PolicyValueCudaContext {
    pub fn new(device_id: usize) -> Result<Self, cudarc::driver::DriverError> {
        let ctx = CudaContext::new(device_id)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| {
            // cublas error を driver error に wrap
            eprintln!("CudaBlas init failed: {e:?}");
            cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN)
        })?;
        Ok(Self { ctx, stream, blas })
    }
}
