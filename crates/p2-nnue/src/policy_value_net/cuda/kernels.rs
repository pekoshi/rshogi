//! 自前 CUDA kernel (NVRTC で実行時 compile)
//!
//! cuBLAS で matvec、 自前で element-wise activation/binop。

/// kernel ソース (NVRTC 入力)
pub const KERNELS_CU: &str = r#"
extern "C" __global__ void screlu_inplace(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        x[i] = v * v;
    }
}

extern "C" __global__ void leaky_relu_inplace(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v > 0.0f ? v : 0.01f * v;
    }
}

extern "C" __global__ void silu_inplace(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

// element-wise mul: out[i] = a[i] * b[i]
extern "C" __global__ void elem_mul(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

// FT sparse sum: out = bias、 indices に対応する weight column を加算
// weight は row-major (L1, INPUT_DIM)、 indices は active feature index 列
extern "C" __global__ void ft_sparse_add(
    const float* weight, const int* indices, int n_indices,
    int l1_size, int input_dim, float* out)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < l1_size) {
        float acc = out[row]; // 事前に bias で初期化されている
        for (int k = 0; k < n_indices; k++) {
            int idx = indices[k];
            acc += weight[row * input_dim + idx];
        }
        out[row] = acc;
    }
}

// set bias (overwrite): x[i] = bias[i]
// (buffer reuse 用、 前回値を上書き)
extern "C" __global__ void add_bias(float* x, const float* bias, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = bias[i];
}

// naive matvec with bias: y[i] = bias[i] + sum_j(w[i*in_dim + j] * x[j])
// w is row-major (out_dim, in_dim)
extern "C" __global__ void matvec_bias(
    const float* w, const float* x, const float* bias, float* y,
    int out_dim, int in_dim)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < out_dim) {
        float acc = bias[row];
        const float* row_w = w + row * in_dim;
        for (int j = 0; j < in_dim; j++) {
            acc += row_w[j] * x[j];
        }
        y[row] = acc;
    }
}

// batch bias broadcast: Y[i, b] = bias[i]  (column-major col-vector broadcast)
// Y is (out_dim, batch_size) column-major
extern "C" __global__ void add_bias_batch(
    float* y, const float* bias, int out_dim, int batch_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < out_dim && col < batch_size) {
        y[col * out_dim + row] = bias[row];
    }
}

// batch SCReLU in-place (elementwise on flat buffer)
extern "C" __global__ void screlu_batch(float* x, int n_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_total) {
        float v = x[i];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        x[i] = v * v;
    }
}

// batch LeakyReLU in-place
extern "C" __global__ void leaky_relu_batch(float* x, int n_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_total) {
        float v = x[i];
        x[i] = v > 0.0f ? v : 0.01f * v;
    }
}

// batch SiLU in-place
extern "C" __global__ void silu_batch(float* x, int n_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_total) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

// batch elementwise mul: out[i] = a[i] * b[i]
extern "C" __global__ void elem_mul_batch(
    const float* a, const float* b, float* out, int n_total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_total) out[i] = a[i] * b[i];
}

// batch FT sparse add: 各 batch sample で独立に sparse sum
// Y is (L1, batch_size) column-major
// indices_offsets[b] = batch b の開始 offset
// indices[]: flat 配列、 batch b の indices = indices[offsets[b]..offsets[b+1]]
extern "C" __global__ void ft_sparse_add_batch(
    const float* weight, const int* indices_flat, const int* offsets,
    int l1_size, int input_dim, int batch_size,
    float* y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    if (row < l1_size && batch < batch_size) {
        int start = offsets[batch];
        int end = offsets[batch + 1];
        float acc = y[batch * l1_size + row];
        for (int k = start; k < end; k++) {
            int idx = indices_flat[k];
            acc += weight[row * input_dim + idx];
        }
        y[batch * l1_size + row] = acc;
    }
}
"#;
