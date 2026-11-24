//! CUDA メモリ管理 RAII ラッパー
//!
//! GPU デバイスメモリとピンドホストメモリを安全に管理する。

use std::ffi::c_void;
use std::marker::PhantomData;

use super::ffi::{self, CUDA_HOST_ALLOC_DEFAULT, CudaMemcpyKind, CudaStreamT, check_cuda};

/// GPU デバイスメモリ RAII ラッパー
pub(crate) struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    /// GPU 上にバッファを確保
    pub(crate) fn new(len: usize) -> Result<Self, String> {
        let size = len * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: cudaMalloc は有効なポインタまたはエラーコードを返す
        check_cuda(unsafe { ffi::cudaMalloc(&mut ptr, size) })?;
        Ok(Self {
            ptr: ptr as *mut T,
            len,
            _marker: PhantomData,
        })
    }

    /// デバイスポインタを取得
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// デバイスポインタを取得 (mutable)
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// バッファサイズ (バイト)
    pub(crate) fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr は cudaMalloc で確保済み
            unsafe {
                let _ = ffi::cudaFree(self.ptr as *mut c_void);
            }
        }
    }
}

// SAFETY: DeviceBuffer は GPU メモリへのポインタを保持するだけで、
// CPU スレッド間で安全に移動・共有できる。
unsafe impl<T> Send for DeviceBuffer<T> {}
unsafe impl<T> Sync for DeviceBuffer<T> {}

/// ピンドホストメモリ RAII ラッパー
///
/// `cudaHostAlloc` で確保されたページロックメモリ。
/// H2D/D2H 非同期転送で DMA を利用するために必要。
pub(crate) struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> PinnedBuffer<T> {
    /// ピンドメモリを確保
    pub(crate) fn new(len: usize) -> Result<Self, String> {
        let size = len * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: cudaHostAlloc は有効なポインタまたはエラーコードを返す
        check_cuda(unsafe { ffi::cudaHostAlloc(&mut ptr, size, CUDA_HOST_ALLOC_DEFAULT) })?;
        Ok(Self {
            ptr: ptr as *mut T,
            len,
            _marker: PhantomData,
        })
    }

    /// ホストポインタを取得
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// ホストポインタを取得 (mutable)
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// スライスとしてアクセス
    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: ptr は cudaHostAlloc で len 要素分確保済み
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// mutable スライスとしてアクセス
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: ptr は cudaHostAlloc で len 要素分確保済み、排他参照保証
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr は cudaHostAlloc で確保済み
            unsafe {
                let _ = ffi::cudaFreeHost(self.ptr as *mut c_void);
            }
        }
    }
}

// SAFETY: PinnedBuffer はページロックメモリへのポインタを保持するだけ。
unsafe impl<T> Send for PinnedBuffer<T> {}
unsafe impl<T> Sync for PinnedBuffer<T> {}

/// CUDA ストリーム RAII ラッパー
pub(crate) struct CudaStream {
    stream: CudaStreamT,
}

impl CudaStream {
    /// 新しいストリームを作成
    pub(crate) fn new() -> Result<Self, String> {
        let mut stream: CudaStreamT = std::ptr::null_mut();
        // SAFETY: cudaStreamCreate は有効なストリームハンドルを返す
        check_cuda(unsafe { ffi::cudaStreamCreate(&mut stream) })?;
        Ok(Self { stream })
    }

    /// ストリームハンドルを取得
    pub(crate) fn handle(&self) -> CudaStreamT {
        self.stream
    }

    /// ストリームの完了を待機
    pub(crate) fn synchronize(&self) -> Result<(), String> {
        // SAFETY: stream は cudaStreamCreate で作成済み
        check_cuda(unsafe { ffi::cudaStreamSynchronize(self.stream) })
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            // SAFETY: stream は cudaStreamCreate で作成済み
            unsafe {
                let _ = ffi::cudaStreamDestroy(self.stream);
            }
        }
    }
}

/// 非同期 Host→Device メモリコピー
pub(crate) fn memcpy_h2d_async<T>(
    dst: &mut DeviceBuffer<T>,
    src: &PinnedBuffer<T>,
    count: usize,
    stream: &CudaStream,
) -> Result<(), String> {
    let size = count * std::mem::size_of::<T>();
    debug_assert!(size <= dst.size_bytes());
    // SAFETY: dst/src は有効なバッファ、count 要素分のサイズが収まることを確認済み
    check_cuda(unsafe {
        ffi::cudaMemcpyAsync(
            dst.as_mut_ptr() as *mut c_void,
            src.as_ptr() as *const c_void,
            size,
            CudaMemcpyKind::HostToDevice,
            stream.handle(),
        )
    })
}

/// 非同期 Device→Host メモリコピー
pub(crate) fn memcpy_d2h_async<T>(
    dst: &mut PinnedBuffer<T>,
    src: &DeviceBuffer<T>,
    count: usize,
    stream: &CudaStream,
) -> Result<(), String> {
    let size = count * std::mem::size_of::<T>();
    // SAFETY: dst/src は有効なバッファ、count 要素分のサイズが収まることを確認済み
    check_cuda(unsafe {
        ffi::cudaMemcpyAsync(
            dst.as_mut_ptr() as *mut c_void,
            src.as_ptr() as *const c_void,
            size,
            CudaMemcpyKind::DeviceToHost,
            stream.handle(),
        )
    })
}
