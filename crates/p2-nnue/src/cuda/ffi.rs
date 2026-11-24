//! CUDA FFI 宣言
//!
//! build.rs でコンパイルされた kernels.cu の C 関数と
//! CUDA Runtime API のバインディング。

use std::ffi::c_void;
use std::os::raw::c_int;

/// CUDA ストリームハンドル
pub(crate) type CudaStreamT = *mut c_void;

/// CUDA メモリコピー方向
#[repr(C)]
#[allow(dead_code)]
pub(crate) enum CudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

// ─── CUDA Runtime API ───

extern "C" {
    pub(crate) fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
    pub(crate) fn cudaFree(dev_ptr: *mut c_void) -> c_int;
    pub(crate) fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    pub(crate) fn cudaFreeHost(ptr: *mut c_void) -> c_int;
    pub(crate) fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: CudaMemcpyKind,
        stream: CudaStreamT,
    ) -> c_int;
    pub(crate) fn cudaStreamCreate(stream: *mut CudaStreamT) -> c_int;
    pub(crate) fn cudaStreamDestroy(stream: CudaStreamT) -> c_int;
    pub(crate) fn cudaStreamSynchronize(stream: CudaStreamT) -> c_int;
    pub(crate) fn cudaGetLastError() -> c_int;
    pub(crate) fn cudaGetErrorString(error: c_int) -> *const std::ffi::c_char;
}

use std::os::raw::c_uint;

/// ピンドメモリ割り当てフラグ
pub(crate) const CUDA_HOST_ALLOC_DEFAULT: c_uint = 0;

// ─── P2 NNUE カーネル (kernels.cu) ───

extern "C" {
    /// Layer 1 forward: i32[batch][2048] × i16[512][2048] → i32[batch][512]
    pub(crate) fn p2_layer1_forward(
        input: *const i32,
        weights: *const i16,
        biases: *const i32,
        output: *mut i32,
        batch_size: c_int,
        input_dim: c_int,
        output_dim: c_int,
    );

    /// Layer 2 forward: u8[batch][512] × i8[256][512] → i32[batch][256]
    pub(crate) fn p2_layer2_forward(
        input: *const u8,
        weights: *const i8,
        biases: *const i32,
        output: *mut i32,
        batch_size: c_int,
        input_dim: c_int,
        output_dim: c_int,
    );

    /// Layer 3 forward: u8[batch][256] × i8[1][256] → i32[batch]
    pub(crate) fn p2_layer3_forward(
        input: *const u8,
        weights: *const i8,
        biases: *const i32,
        output: *mut i32,
        batch_size: c_int,
        input_dim: c_int,
    );

    /// ClippedReLU: i32[n] → u8[n], shift >> WEIGHT_SCALE_BITS, clamp [0, 127]
    pub(crate) fn p2_clipped_relu(input: *const i32, output: *mut u8, n: c_int, shift: c_int);
}

/// CUDA エラーチェック
///
/// 戻り値が 0 (cudaSuccess) でなければエラーメッセージを返す。
pub(crate) fn check_cuda(code: c_int) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        // SAFETY: cudaGetErrorString は有効な null-terminated 文字列を返す
        let msg = unsafe {
            let ptr = cudaGetErrorString(code);
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        };
        Err(format!("CUDA error {code}: {msg}"))
    }
}
