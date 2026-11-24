//! Policy-Value GPU 推論 (cudarc 直接利用、 feature `pv-cuda`)
//!
//! 1050 Ti で dlshogi 互角ライン (1500-3000 leaf/sec) 到達目標。
//!
//! 構成:
//! - device.rs: CudaContext + CudaStream + cuBLAS 初期化
//! - kernels.rs: NVRTC で自前 element-wise kernel compile
//! - network.rs: PolicyValueNetCuda (GPU forward)
//! - evaluator.rs: PolicyValueEvaluatorCuda (Position → forward)

pub mod device;
pub mod kernels;
pub mod network;

pub use device::PolicyValueCudaContext;
pub use network::PolicyValueNetCuda;
