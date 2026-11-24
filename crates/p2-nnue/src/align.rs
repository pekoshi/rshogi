//! 64B アライメントユーティリティ
//!
//! NNUE 層の重み・中間バッファを 64B (キャッシュライン) 境界に配置する。

use std::alloc::{Layout, alloc_zeroed, dealloc};

/// スタック上の 64B アライン配列ラッパー
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Aligned<T: Copy>(pub T);

/// ヒープ上の 64B アライン動的配列
pub struct AlignedBox<T: Copy> {
    ptr: *mut T,
    len: usize,
    layout: Layout,
}

impl<T: Copy> AlignedBox<T> {
    /// 指定長のゼロ初期化済みアライン配列を確保する
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                layout: Layout::from_size_align(0, 64).unwrap(),
            };
        }
        let size = std::mem::size_of::<T>() * len;
        let layout = Layout::from_size_align(size, 64).unwrap();
        // SAFETY: layout は非ゼロサイズ、64B アラインで有効。
        // alloc_zeroed はゼロ初期化済みメモリを返す。
        // T: Copy なのでゼロビットパターンは安全（i8, i16, i32, u8 等の数値型）。
        let ptr = unsafe { alloc_zeroed(layout) as *mut T };
        assert!(!ptr.is_null(), "メモリ確保失敗");
        Self { ptr, len, layout }
    }
}

impl<T: Copy> std::ops::Deref for AlignedBox<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: ptr は new() で確保済み、len 要素分有効。
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Copy> std::ops::DerefMut for AlignedBox<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: ptr は new() で確保済み、len 要素分有効、排他参照保証。
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Copy> Drop for AlignedBox<T> {
    fn drop(&mut self) {
        if self.layout.size() > 0 {
            // SAFETY: ptr は同一 layout で alloc_zeroed から取得済み。
            unsafe {
                dealloc(self.ptr as *mut u8, self.layout);
            }
        }
    }
}

// SAFETY: T: Copy なので Send/Sync は安全。
// AlignedBox は排他的にメモリを所有しており、他スレッドとの共有は &self 経由のみ。
unsafe impl<T: Copy + Send> Send for AlignedBox<T> {}
unsafe impl<T: Copy + Sync> Sync for AlignedBox<T> {}

impl<T: Copy> Clone for AlignedBox<T> {
    fn clone(&self) -> Self {
        let mut new_box = Self::new(self.len);
        new_box.copy_from_slice(self);
        new_box
    }
}
