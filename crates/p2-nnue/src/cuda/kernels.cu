/**
 * P2 NNUE CUDA カーネル
 *
 * ネットワーク: 2048→512→256→1
 * バッチ推論用。各カーネルは batch_size パラメータでバッチ全体を処理する。
 */

#include <cstdint>

// ─── Layer 1: i32[batch][2048] × i16[512][2048] → i32[batch][512] ───

/**
 * 各スレッドブロックが 1 出力ニューロンを担当。
 * shared memory で部分和を集約する。
 *
 * grid:  (output_dim, batch_size)
 * block: (256)
 */
extern "C" __global__ void p2_layer1_forward(
    const int32_t* __restrict__ input,
    const int16_t* __restrict__ weights,
    const int32_t* __restrict__ biases,
    int32_t* __restrict__ output,
    int batch_size,
    int input_dim,
    int output_dim)
{
    const int out_idx = blockIdx.x;   // 出力ニューロン index
    const int batch = blockIdx.y;     // バッチ index

    if (batch >= batch_size || out_idx >= output_dim) return;

    __shared__ int32_t shared[256];

    const int32_t* in_row = input + batch * input_dim;
    const int16_t* w_row = weights + out_idx * input_dim;

    // 各スレッドが input_dim/blockDim.x 要素を担当
    int32_t acc = 0;
    for (int i = threadIdx.x; i < input_dim; i += blockDim.x) {
        acc += (int32_t)w_row[i] * in_row[i];
    }

    shared[threadIdx.x] = acc;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[batch * output_dim + out_idx] = biases[out_idx] + shared[0];
    }
}

// ─── Layer 2: u8[batch][512] × i8[256][512] → i32[batch][256] ───

/**
 * dp4a (INT8 dot product) を使用。
 * 各スレッドブロックが 1 出力ニューロンを担当。
 *
 * grid:  (output_dim, batch_size)
 * block: (128)
 */
extern "C" __global__ void p2_layer2_forward(
    const uint8_t* __restrict__ input,
    const int8_t* __restrict__ weights,
    const int32_t* __restrict__ biases,
    int32_t* __restrict__ output,
    int batch_size,
    int input_dim,
    int output_dim)
{
    const int out_idx = blockIdx.x;
    const int batch = blockIdx.y;

    if (batch >= batch_size || out_idx >= output_dim) return;

    __shared__ int32_t shared[128];

    const uint8_t* in_row = input + batch * input_dim;
    const int8_t* w_row = weights + out_idx * input_dim;

    // 4 バイト単位で dp4a 演算
    const int32_t* in32 = (const int32_t*)in_row;
    const int32_t* w32 = (const int32_t*)w_row;
    const int chunks = input_dim / 4;

    int32_t acc = 0;
    for (int i = threadIdx.x; i < chunks; i += blockDim.x) {
#if __CUDA_ARCH__ >= 610
        acc = __dp4a(in32[i], w32[i], acc);
#else
        // dp4a エミュレーション (Compute Capability < 6.1)
        const uint8_t* a = (const uint8_t*)&in32[i];
        const int8_t* b = (const int8_t*)&w32[i];
        acc += (int32_t)a[0] * b[0] + (int32_t)a[1] * b[1] +
               (int32_t)a[2] * b[2] + (int32_t)a[3] * b[3];
#endif
    }

    shared[threadIdx.x] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[batch * output_dim + out_idx] = biases[out_idx] + shared[0];
    }
}

// ─── Layer 3: u8[batch][256] × i8[1][256] → i32[batch] ───

/**
 * 各ブロックが 1 バッチ要素を担当。
 *
 * grid:  (batch_size)
 * block: (128)
 */
extern "C" __global__ void p2_layer3_forward(
    const uint8_t* __restrict__ input,
    const int8_t* __restrict__ weights,
    const int32_t* __restrict__ biases,
    int32_t* __restrict__ output,
    int batch_size,
    int input_dim)
{
    const int batch = blockIdx.x;
    if (batch >= batch_size) return;

    __shared__ int32_t shared[128];

    const uint8_t* in_row = input + batch * input_dim;

    int32_t acc = 0;
    for (int i = threadIdx.x; i < input_dim; i += blockDim.x) {
        acc += (int32_t)in_row[i] * (int32_t)weights[i];
    }

    shared[threadIdx.x] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[batch] = biases[0] + shared[0];
    }
}

// ─── ClippedReLU: i32[n] → u8[n] ───

/**
 * element-wise: output[i] = clamp(input[i] >> shift, 0, 127)
 *
 * grid:  ceil(n / 256)
 * block: (256)
 */
extern "C" __global__ void p2_clipped_relu(
    const int32_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int n,
    int shift)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int32_t val = input[idx] >> shift;
    if (val < 0) val = 0;
    if (val > 127) val = 127;
    output[idx] = (uint8_t)val;
}
