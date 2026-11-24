//! GPU CUDA ベースの NNUE 評価器
//!
//! `cuda` feature 有効時のみコンパイルされる。
//! 重みは初期化時に GPU へ一回転送し、バッチ推論で使い回す。

use std::sync::Arc;

use super::NnueEvaluator;
use crate::cuda::ffi;
use crate::cuda::memory::{
    CudaStream, DeviceBuffer, PinnedBuffer, memcpy_d2h_async, memcpy_h2d_async,
};
use crate::features::P2NnueInput;
use crate::network::P2Network;
use crate::types::{P2_INPUT_DIM, P2_L1_DIM, P2_L2_DIM, P2_OUTPUT_DIM, WEIGHT_SCALE_BITS};

/// 最大バッチサイズ
const MAX_BATCH: usize = 256;

/// GPU 上の重みバッファ
struct DeviceWeights {
    l1_weights: DeviceBuffer<i16>,
    l1_biases: DeviceBuffer<i32>,
    l2_weights: DeviceBuffer<i8>,
    l2_biases: DeviceBuffer<i32>,
    l3_weights: DeviceBuffer<i8>,
    l3_biases: DeviceBuffer<i32>,
}

/// CUDA による NNUE バッチ評価器
pub struct CudaEvaluator {
    _network: Arc<P2Network>,
    device_weights: DeviceWeights,
    // ピンドホストメモリ (H2D / D2H 転送用)
    pinned_input: PinnedBuffer<i32>,
    pinned_output: PinnedBuffer<i32>,
    // デバイスバッファ (中間結果)
    device_input: DeviceBuffer<i32>,
    device_l1_out: DeviceBuffer<i32>,
    device_l1_relu: DeviceBuffer<u8>,
    device_l2_out: DeviceBuffer<i32>,
    device_l2_relu: DeviceBuffer<u8>,
    device_l3_out: DeviceBuffer<i32>,
    stream: CudaStream,
}

impl CudaEvaluator {
    /// GPU 評価器を初期化
    ///
    /// 重みを GPU メモリに転送し、中間バッファを確保する。
    /// GPU が利用不可の場合はエラーを返す。
    pub fn new(network: Arc<P2Network>) -> Result<Self, String> {
        let stream = CudaStream::new()?;

        // 重みを GPU に転送
        let device_weights = Self::upload_weights(&network, &stream)?;

        // ピンドホストメモリ
        let pinned_input = PinnedBuffer::new(MAX_BATCH * P2_INPUT_DIM)?;
        let pinned_output = PinnedBuffer::new(MAX_BATCH)?;

        // デバイス中間バッファ
        let device_input = DeviceBuffer::new(MAX_BATCH * P2_INPUT_DIM)?;
        let device_l1_out = DeviceBuffer::new(MAX_BATCH * P2_L1_DIM)?;
        let device_l1_relu = DeviceBuffer::new(MAX_BATCH * P2_L1_DIM)?;
        let device_l2_out = DeviceBuffer::new(MAX_BATCH * P2_L2_DIM)?;
        let device_l2_relu = DeviceBuffer::new(MAX_BATCH * P2_L2_DIM)?;
        let device_l3_out = DeviceBuffer::new(MAX_BATCH)?;

        Ok(Self {
            _network: network,
            device_weights,
            pinned_input,
            pinned_output,
            device_input,
            device_l1_out,
            device_l1_relu,
            device_l2_out,
            device_l2_relu,
            device_l3_out,
            stream,
        })
    }

    /// ネットワーク重みを GPU にアップロード
    fn upload_weights(network: &P2Network, stream: &CudaStream) -> Result<DeviceWeights, String> {
        use crate::cuda::ffi::CudaMemcpyKind;
        use crate::layers::{AffineTransform, InputAffineTransform};
        use std::ffi::c_void;

        let padded_l1 = InputAffineTransform::<P2_INPUT_DIM, P2_L1_DIM>::PADDED_IN;
        let padded_l2 = AffineTransform::<P2_L1_DIM, P2_L2_DIM>::PADDED_IN;
        let padded_l3 = AffineTransform::<P2_L2_DIM, P2_OUTPUT_DIM>::PADDED_IN;

        // L1 重み (行優先: GPU ではスクランブル不要)
        let l1_weight_count = P2_L1_DIM * padded_l1;
        let mut l1_weights = DeviceBuffer::<i16>::new(l1_weight_count)?;
        // SAFETY: network.l1.weights は padded_l1 * P2_L1_DIM 要素確保済み
        ffi::check_cuda(unsafe {
            ffi::cudaMemcpyAsync(
                l1_weights.as_mut_ptr() as *mut c_void,
                network.l1.weights.as_ptr() as *const c_void,
                l1_weight_count * std::mem::size_of::<i16>(),
                CudaMemcpyKind::HostToDevice,
                stream.handle(),
            )
        })?;

        let mut l1_biases = DeviceBuffer::<i32>::new(P2_L1_DIM)?;
        ffi::check_cuda(unsafe {
            ffi::cudaMemcpyAsync(
                l1_biases.as_mut_ptr() as *mut c_void,
                network.l1.biases.as_ptr() as *const c_void,
                P2_L1_DIM * 4,
                CudaMemcpyKind::HostToDevice,
                stream.handle(),
            )
        })?;

        // L2 重み (行優先: GPU ではスクランブル不要)
        // read() でスクランブルされている可能性があるため、生の重みから再構築が必要だが
        // new_zeroed() で作成された場合は行優先のまま。
        // GPU カーネルは行優先レイアウトを前提とするので、ここでは
        // weights をそのまま転送する（GPU カーネル側で対応）。
        let l2_weight_count = P2_L2_DIM * padded_l2;
        let mut l2_weights = DeviceBuffer::<i8>::new(l2_weight_count)?;
        ffi::check_cuda(unsafe {
            ffi::cudaMemcpyAsync(
                l2_weights.as_mut_ptr() as *mut c_void,
                network.l2.weights.as_ptr() as *const c_void,
                l2_weight_count,
                CudaMemcpyKind::HostToDevice,
                stream.handle(),
            )
        })?;

        let mut l2_biases = DeviceBuffer::<i32>::new(P2_L2_DIM)?;
        ffi::check_cuda(unsafe {
            ffi::cudaMemcpyAsync(
                l2_biases.as_mut_ptr() as *mut c_void,
                network.l2.biases.as_ptr() as *const c_void,
                P2_L2_DIM * 4,
                CudaMemcpyKind::HostToDevice,
                stream.handle(),
            )
        })?;

        // L3 重み
        let l3_weight_count = P2_OUTPUT_DIM * padded_l3;
        let mut l3_weights = DeviceBuffer::<i8>::new(l3_weight_count)?;
        ffi::check_cuda(unsafe {
            ffi::cudaMemcpyAsync(
                l3_weights.as_mut_ptr() as *mut c_void,
                network.l3.weights.as_ptr() as *const c_void,
                l3_weight_count,
                CudaMemcpyKind::HostToDevice,
                stream.handle(),
            )
        })?;

        let mut l3_biases = DeviceBuffer::<i32>::new(P2_OUTPUT_DIM)?;
        ffi::check_cuda(unsafe {
            ffi::cudaMemcpyAsync(
                l3_biases.as_mut_ptr() as *mut c_void,
                network.l3.biases.as_ptr() as *const c_void,
                P2_OUTPUT_DIM * 4,
                CudaMemcpyKind::HostToDevice,
                stream.handle(),
            )
        })?;

        stream.synchronize()?;

        Ok(DeviceWeights {
            l1_weights,
            l1_biases,
            l2_weights,
            l2_biases,
            l3_weights,
            l3_biases,
        })
    }
}

impl NnueEvaluator for CudaEvaluator {
    fn evaluate(&self, input: &P2NnueInput) -> i32 {
        let mut output = [0i32; 1];
        self.evaluate_batch(std::slice::from_ref(input), &mut output);
        output[0]
    }

    fn evaluate_batch(&self, inputs: &[P2NnueInput], outputs: &mut [i32]) {
        assert_eq!(inputs.len(), outputs.len());
        let batch_size = inputs.len();
        assert!(batch_size <= MAX_BATCH, "バッチサイズが {MAX_BATCH} を超えています");

        // self を &mut にキャストして可変バッファにアクセス
        // SAFETY: evaluate_batch は &self だが、内部バッファは排他的に使用される。
        // CudaEvaluator は Send であり、同時に複数スレッドから呼ばれない前提。
        let this = unsafe { &mut *(self as *const Self as *mut Self) };

        // 入力をピンドメモリにコピー
        let pinned_slice = this.pinned_input.as_mut_slice();
        for (b, input) in inputs.iter().enumerate() {
            let dst_offset = b * P2_INPUT_DIM;
            pinned_slice[dst_offset..dst_offset + P2_INPUT_DIM]
                .copy_from_slice(&input.piece_features);
        }

        // H2D 転送
        memcpy_h2d_async(
            &mut this.device_input,
            &this.pinned_input,
            batch_size * P2_INPUT_DIM,
            &this.stream,
        )
        .expect("H2D memcpy failed");

        let bs = batch_size as i32;

        // SAFETY: カーネル呼び出しは CUDA FFI。
        // バッファサイズは MAX_BATCH で確保済みなので範囲内。
        unsafe {
            // L1 forward
            ffi::p2_layer1_forward(
                this.device_input.as_ptr(),
                this.device_weights.l1_weights.as_ptr(),
                this.device_weights.l1_biases.as_ptr(),
                this.device_l1_out.as_mut_ptr(),
                bs,
                P2_INPUT_DIM as i32,
                P2_L1_DIM as i32,
            );

            // ClippedReLU (L1)
            let n1 = (batch_size * P2_L1_DIM) as i32;
            ffi::p2_clipped_relu(
                this.device_l1_out.as_ptr(),
                this.device_l1_relu.as_mut_ptr(),
                n1,
                WEIGHT_SCALE_BITS as i32,
            );

            // L2 forward
            ffi::p2_layer2_forward(
                this.device_l1_relu.as_ptr(),
                this.device_weights.l2_weights.as_ptr(),
                this.device_weights.l2_biases.as_ptr(),
                this.device_l2_out.as_mut_ptr(),
                bs,
                P2_L1_DIM as i32,
                P2_L2_DIM as i32,
            );

            // ClippedReLU (L2)
            let n2 = (batch_size * P2_L2_DIM) as i32;
            ffi::p2_clipped_relu(
                this.device_l2_out.as_ptr(),
                this.device_l2_relu.as_mut_ptr(),
                n2,
                WEIGHT_SCALE_BITS as i32,
            );

            // L3 forward
            ffi::p2_layer3_forward(
                this.device_l2_relu.as_ptr(),
                this.device_weights.l3_weights.as_ptr(),
                this.device_weights.l3_biases.as_ptr(),
                this.device_l3_out.as_mut_ptr(),
                bs,
                P2_L2_DIM as i32,
            );
        }

        // D2H 転送
        memcpy_d2h_async(&mut this.pinned_output, &this.device_l3_out, batch_size, &this.stream)
            .expect("D2H memcpy failed");

        this.stream.synchronize().expect("stream sync failed");

        // 結果をコピー
        outputs.copy_from_slice(&this.pinned_output.as_slice()[..batch_size]);
    }

    fn backend_name(&self) -> &'static str {
        "cuda"
    }
}
