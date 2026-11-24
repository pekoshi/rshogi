//! PolicyValueNet GPU 推論 (naive CUDA matvec + element-wise kernel)

use std::sync::Arc;

use cudarc::cublas::{Gemm, GemmConfig, Gemv, GemvConfig, sys::cublasOperation_t};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use super::device::PolicyValueCudaContext;
use super::kernels::KERNELS_CU;
use crate::policy_value_net::config::{
    EVAL_SCALE, INPUT_DIM, L1_SIZE, L2_SIZE, L3_SIZE, NormalisationConfig, POLICY_DIMS,
    PolicyValueConfig,
};
use crate::policy_value_net::loader::PolicyValueWeights;
use crate::policy_value_net::network::PolicyValueOutput;

pub struct PolicyValueCudaWeights {
    pub cfg: PolicyValueConfig,
    pub l0w: CudaSlice<f32>,
    pub l0b: CudaSlice<f32>,
    pub l1w: CudaSlice<f32>,
    pub l1b: CudaSlice<f32>,
    pub l2w: CudaSlice<f32>,
    pub l2b: CudaSlice<f32>,
    pub v_outw: CudaSlice<f32>,
    pub v_outb: CudaSlice<f32>,
    pub pol_w1w: CudaSlice<f32>,
    pub pol_w1b: CudaSlice<f32>,
    pub pol_w3w: CudaSlice<f32>,
    pub pol_w3b: CudaSlice<f32>,
    pub pol_ow: CudaSlice<f32>,
    pub pol_ob: CudaSlice<f32>,
}

pub struct PolicyValueNetCuda {
    pub ctx: Arc<PolicyValueCudaContext>,
    pub weights: PolicyValueCudaWeights,
    pub module: Arc<CudaModule>,
    // kernel functions
    fn_screlu: CudaFunction,
    fn_leaky_relu: CudaFunction,
    fn_silu: CudaFunction,
    fn_elem_mul: CudaFunction,
    fn_ft_sparse_add: CudaFunction,
    fn_add_bias: CudaFunction,
    fn_matvec_bias: CudaFunction,
    // batch kernels
    fn_add_bias_batch: CudaFunction,
    fn_screlu_batch: CudaFunction,
    fn_leaky_relu_batch: CudaFunction,
    fn_silu_batch: CudaFunction,
    fn_elem_mul_batch: CudaFunction,
    fn_ft_sparse_add_batch: CudaFunction,
    // Working buffers (alloc once、 forward で reuse)
    buf_stm_acc: CudaSlice<f32>,
    buf_nstm_acc: CudaSlice<f32>,
    buf_concat: CudaSlice<f32>,
    buf_h1: CudaSlice<f32>,
    buf_h2: CudaSlice<f32>,
    buf_value_out: CudaSlice<f32>,
    buf_gate: CudaSlice<f32>,
    buf_value_pre: CudaSlice<f32>,
    buf_swiglu: CudaSlice<f32>,
    buf_policy_logits: CudaSlice<f32>,
    buf_stm_idx: CudaSlice<i32>,
    buf_nstm_idx: CudaSlice<i32>,
}

#[inline]
fn launch_cfg(n: usize, threads: u32) -> LaunchConfig {
    let blocks = (n as u32).div_ceil(threads);
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// cuBLAS gemv で y += W·x を実行
/// W は row-major (out_dim, in_dim) で配置 = column-major で (in_dim, out_dim)
/// → trans=T で op(A) = (out_dim, in_dim)、 出力 dim = out_dim
fn gemv_with_bias(
    blas: &cudarc::cublas::CudaBlas,
    w: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
) -> Result<(), cudarc::cublas::result::CublasError> {
    let cfg = GemvConfig {
        trans: cublasOperation_t::CUBLAS_OP_T,
        m: in_dim as i32,
        n: out_dim as i32,
        alpha: 1.0f32,
        lda: in_dim as i32,
        incx: 1,
        beta: 1.0f32,
        incy: 1,
    };
    unsafe { blas.gemv(cfg, w, x, y) }
}

/// cuBLAS gemm で Y (out, batch) = W·X (in, batch) + bias (broadcast 事前)
/// W row-major (out, in)、 X col-major (in, batch)、 Y col-major (out, batch)
fn gemm_with_bias_batch(
    blas: &cudarc::cublas::CudaBlas,
    w: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    out_dim: usize,
    in_dim: usize,
    batch_size: usize,
) -> Result<(), cudarc::cublas::result::CublasError> {
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: out_dim as i32,
        n: batch_size as i32,
        k: in_dim as i32,
        alpha: 1.0f32,
        lda: in_dim as i32,
        ldb: in_dim as i32,
        beta: 1.0f32,
        ldc: out_dim as i32,
    };
    unsafe { blas.gemm(cfg, w, x, y) }
}

impl PolicyValueNetCuda {
    pub fn new(
        ctx: Arc<PolicyValueCudaContext>,
        cpu_weights: &PolicyValueWeights,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = &ctx.stream;
        let weights = PolicyValueCudaWeights {
            cfg: cpu_weights.cfg,
            l0w: stream.memcpy_stod(&cpu_weights.l0w)?,
            l0b: stream.memcpy_stod(&cpu_weights.l0b)?,
            l1w: stream.memcpy_stod(&cpu_weights.l1w)?,
            l1b: stream.memcpy_stod(&cpu_weights.l1b)?,
            l2w: stream.memcpy_stod(&cpu_weights.l2w)?,
            l2b: stream.memcpy_stod(&cpu_weights.l2b)?,
            v_outw: stream.memcpy_stod(&cpu_weights.v_outw)?,
            v_outb: stream.memcpy_stod(&cpu_weights.v_outb)?,
            pol_w1w: stream.memcpy_stod(&cpu_weights.pol_w1w)?,
            pol_w1b: stream.memcpy_stod(&cpu_weights.pol_w1b)?,
            pol_w3w: stream.memcpy_stod(&cpu_weights.pol_w3w)?,
            pol_w3b: stream.memcpy_stod(&cpu_weights.pol_w3b)?,
            pol_ow: stream.memcpy_stod(&cpu_weights.pol_ow)?,
            pol_ob: stream.memcpy_stod(&cpu_weights.pol_ob)?,
        };

        let ptx = compile_ptx(KERNELS_CU)?;
        let module = ctx.ctx.load_module(ptx)?;

        let fn_screlu = module.load_function("screlu_inplace")?;
        let fn_leaky_relu = module.load_function("leaky_relu_inplace")?;
        let fn_silu = module.load_function("silu_inplace")?;
        let fn_elem_mul = module.load_function("elem_mul")?;
        let fn_ft_sparse_add = module.load_function("ft_sparse_add")?;
        let fn_add_bias = module.load_function("add_bias")?;
        let fn_matvec_bias = module.load_function("matvec_bias")?;
        let fn_add_bias_batch = module.load_function("add_bias_batch")?;
        let fn_screlu_batch = module.load_function("screlu_batch")?;
        let fn_leaky_relu_batch = module.load_function("leaky_relu_batch")?;
        let fn_silu_batch = module.load_function("silu_batch")?;
        let fn_elem_mul_batch = module.load_function("elem_mul_batch")?;
        let fn_ft_sparse_add_batch = module.load_function("ft_sparse_add_batch")?;

        // pre-alloc working buffers (forward で reuse)
        let ph = weights.cfg.policy_hidden;
        let buf_stm_acc = stream.alloc_zeros::<f32>(L1_SIZE)?;
        let buf_nstm_acc = stream.alloc_zeros::<f32>(L1_SIZE)?;
        let buf_concat = stream.alloc_zeros::<f32>(2 * L1_SIZE)?;
        let buf_h1 = stream.alloc_zeros::<f32>(L2_SIZE)?;
        let buf_h2 = stream.alloc_zeros::<f32>(L3_SIZE)?;
        let buf_value_out = stream.alloc_zeros::<f32>(1)?;
        let buf_gate = stream.alloc_zeros::<f32>(ph)?;
        let buf_value_pre = stream.alloc_zeros::<f32>(ph)?;
        let buf_swiglu = stream.alloc_zeros::<f32>(ph)?;
        let buf_policy_logits = stream.alloc_zeros::<f32>(POLICY_DIMS)?;
        // 60 = MAX_ACTIVE_FEATURES (再 alloc 不要)
        let buf_stm_idx = stream.alloc_zeros::<i32>(60)?;
        let buf_nstm_idx = stream.alloc_zeros::<i32>(60)?;

        Ok(Self {
            ctx,
            weights,
            module,
            fn_screlu,
            fn_leaky_relu,
            fn_silu,
            fn_elem_mul,
            fn_ft_sparse_add,
            fn_add_bias,
            fn_matvec_bias,
            fn_add_bias_batch,
            fn_screlu_batch,
            fn_leaky_relu_batch,
            fn_silu_batch,
            fn_elem_mul_batch,
            fn_ft_sparse_add_batch,
            buf_stm_acc,
            buf_nstm_acc,
            buf_concat,
            buf_h1,
            buf_h2,
            buf_value_out,
            buf_gate,
            buf_value_pre,
            buf_swiglu,
            buf_policy_logits,
            buf_stm_idx,
            buf_nstm_idx,
        })
    }

    /// y を bias で初期化してから cuBLAS gemv で y += W·x
    fn matvec_bias_inplace(
        ctx: &PolicyValueCudaContext,
        fn_add_bias: &CudaFunction,
        w: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let stream = &ctx.stream;
        unsafe {
            stream
                .launch_builder(fn_add_bias)
                .arg(&mut *y)
                .arg(bias)
                .arg(&(out_dim as i32))
                .launch(launch_cfg(out_dim, 256))?;
        }
        gemv_with_bias(&ctx.blas, w, x, y, out_dim, in_dim)?;
        Ok(())
    }

    /// GPU forward: HalfKAv2_hm active indices → value_raw + policy_logits (buffer reuse)
    pub fn forward(
        &mut self,
        stm_active: &[u32],
        nstm_active: &[u32],
    ) -> Result<PolicyValueOutput, Box<dyn std::error::Error>> {
        // split borrow: ctx と各 buffer を独立 borrow するため
        let Self {
            ctx,
            weights,
            module: _,
            fn_screlu,
            fn_leaky_relu,
            fn_silu,
            fn_elem_mul,
            fn_ft_sparse_add,
            fn_add_bias,
            fn_matvec_bias,
            buf_stm_acc,
            buf_nstm_acc,
            buf_concat,
            buf_h1,
            buf_h2,
            buf_value_out,
            buf_gate,
            buf_value_pre,
            buf_swiglu,
            buf_policy_logits,
            buf_stm_idx,
            buf_nstm_idx,
            ..
        } = self;
        let stream = &ctx.stream;
        let cfg = weights.cfg;
        let ph = cfg.policy_hidden;

        // ── FT (sparse sum) ──
        // indices を pre-alloc buffer に転送
        let stm_indices_i32: Vec<i32> = stm_active.iter().map(|&i| i as i32).collect();
        let nstm_indices_i32: Vec<i32> = nstm_active.iter().map(|&i| i as i32).collect();
        stream.memcpy_htod(&stm_indices_i32, &mut buf_stm_idx.slice_mut(0..stm_active.len()))?;
        stream.memcpy_htod(&nstm_indices_i32, &mut buf_nstm_idx.slice_mut(0..nstm_active.len()))?;

        // bias で初期化 (前回の forward 結果を上書き)
        unsafe {
            stream
                .launch_builder(fn_add_bias)
                .arg(&mut *buf_stm_acc)
                .arg(&weights.l0b)
                .arg(&(L1_SIZE as i32))
                .launch(launch_cfg(L1_SIZE, 256))?;
            stream
                .launch_builder(fn_add_bias)
                .arg(&mut *buf_nstm_acc)
                .arg(&weights.l0b)
                .arg(&(L1_SIZE as i32))
                .launch(launch_cfg(L1_SIZE, 256))?;
        }

        // FT sparse add
        unsafe {
            stream
                .launch_builder(fn_ft_sparse_add)
                .arg(&weights.l0w)
                .arg(&buf_stm_idx.slice(0..stm_active.len()))
                .arg(&(stm_active.len() as i32))
                .arg(&(L1_SIZE as i32))
                .arg(&(INPUT_DIM as i32))
                .arg(&mut *buf_stm_acc)
                .launch(launch_cfg(L1_SIZE, 256))?;
            stream
                .launch_builder(fn_ft_sparse_add)
                .arg(&weights.l0w)
                .arg(&buf_nstm_idx.slice(0..nstm_active.len()))
                .arg(&(nstm_active.len() as i32))
                .arg(&(L1_SIZE as i32))
                .arg(&(INPUT_DIM as i32))
                .arg(&mut *buf_nstm_acc)
                .launch(launch_cfg(L1_SIZE, 256))?;
        }

        // SCReLU
        unsafe {
            stream
                .launch_builder(fn_screlu)
                .arg(&mut *buf_stm_acc)
                .arg(&(L1_SIZE as i32))
                .launch(launch_cfg(L1_SIZE, 256))?;
            stream
                .launch_builder(fn_screlu)
                .arg(&mut *buf_nstm_acc)
                .arg(&(L1_SIZE as i32))
                .launch(launch_cfg(L1_SIZE, 256))?;
        }

        debug_assert!(cfg.normalisation == NormalisationConfig::None, "RMSNorm GPU は未実装");

        // concat: stm_acc + nstm_acc → buf_concat (reuse)
        stream.memcpy_dtod(buf_stm_acc, &mut buf_concat.slice_mut(0..L1_SIZE))?;
        stream.memcpy_dtod(buf_nstm_acc, &mut buf_concat.slice_mut(L1_SIZE..2 * L1_SIZE))?;

        // ── value head (cuBLAS gemv、 buffer reuse) ──
        Self::matvec_bias_inplace(
            ctx,
            fn_add_bias,
            &weights.l1w,
            buf_concat,
            &weights.l1b,
            buf_h1,
            L2_SIZE,
            2 * L1_SIZE,
        )?;
        unsafe {
            stream
                .launch_builder(fn_leaky_relu)
                .arg(&mut *buf_h1)
                .arg(&(L2_SIZE as i32))
                .launch(launch_cfg(L2_SIZE, 32))?;
        }

        Self::matvec_bias_inplace(
            ctx,
            fn_add_bias,
            &weights.l2w,
            buf_h1,
            &weights.l2b,
            buf_h2,
            L3_SIZE,
            L2_SIZE,
        )?;
        unsafe {
            stream
                .launch_builder(fn_leaky_relu)
                .arg(&mut *buf_h2)
                .arg(&(L3_SIZE as i32))
                .launch(launch_cfg(L3_SIZE, 32))?;
        }

        // value_out (1×L3) は自前 kernel
        unsafe {
            stream
                .launch_builder(fn_matvec_bias)
                .arg(&weights.v_outw)
                .arg(&mut *buf_h2)
                .arg(&weights.v_outb)
                .arg(&mut *buf_value_out)
                .arg(&1i32)
                .arg(&(L3_SIZE as i32))
                .launch(launch_cfg(1, 32))?;
        }

        // ── policy head SwiGLU ──
        Self::matvec_bias_inplace(
            ctx,
            fn_add_bias,
            &weights.pol_w1w,
            buf_concat,
            &weights.pol_w1b,
            buf_gate,
            ph,
            2 * L1_SIZE,
        )?;
        Self::matvec_bias_inplace(
            ctx,
            fn_add_bias,
            &weights.pol_w3w,
            buf_concat,
            &weights.pol_w3b,
            buf_value_pre,
            ph,
            2 * L1_SIZE,
        )?;
        unsafe {
            stream
                .launch_builder(fn_silu)
                .arg(&mut *buf_gate)
                .arg(&(ph as i32))
                .launch(launch_cfg(ph, 256))?;
        }

        unsafe {
            stream
                .launch_builder(fn_elem_mul)
                .arg(&*buf_gate)
                .arg(&*buf_value_pre)
                .arg(&mut *buf_swiglu)
                .arg(&(ph as i32))
                .launch(launch_cfg(ph, 256))?;
        }

        Self::matvec_bias_inplace(
            ctx,
            fn_add_bias,
            &weights.pol_ow,
            buf_swiglu,
            &weights.pol_ob,
            buf_policy_logits,
            POLICY_DIMS,
            ph,
        )?;

        // ── CPU download ──
        let value_host = stream.memcpy_dtov(&*buf_value_out)?;
        let policy_host = stream.memcpy_dtov(&*buf_policy_logits)?;
        stream.synchronize()?;

        let _ = (buf_stm_idx, buf_nstm_idx);

        let value_raw = value_host[0];
        let value_cp = (1.0 / (1.0 + (-value_raw).exp())) * EVAL_SCALE;
        Ok(PolicyValueOutput {
            value_raw,
            value_cp,
            policy_logits: policy_host,
        })
    }

    /// batch forward: B 個の leaf を同時 evaluate
    ///
    /// 入力:
    ///   - stm_actives: B 個の active indices (各 Vec<u32>)
    ///   - nstm_actives: 同上
    /// 出力: (value_raws[B], policy_logits flat[B × POLICY_DIMS])
    pub fn forward_batch(
        &self,
        stm_actives: &[Vec<u32>],
        nstm_actives: &[Vec<u32>],
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        assert_eq!(stm_actives.len(), nstm_actives.len());
        let batch = stm_actives.len();
        if batch == 0 {
            return Ok((vec![], vec![]));
        }
        let stream = &self.ctx.stream;
        let blas = &self.ctx.blas;
        let ph = self.weights.cfg.policy_hidden;

        // ── indices flatten + offset table ──
        let mut stm_flat: Vec<i32> = Vec::new();
        let mut stm_offs: Vec<i32> = Vec::with_capacity(batch + 1);
        stm_offs.push(0);
        for v in stm_actives {
            stm_flat.extend(v.iter().map(|&i| i as i32));
            stm_offs.push(stm_flat.len() as i32);
        }
        let mut nstm_flat: Vec<i32> = Vec::new();
        let mut nstm_offs: Vec<i32> = Vec::with_capacity(batch + 1);
        nstm_offs.push(0);
        for v in nstm_actives {
            nstm_flat.extend(v.iter().map(|&i| i as i32));
            nstm_offs.push(nstm_flat.len() as i32);
        }

        let stm_flat_dev: CudaSlice<i32> = stream.memcpy_stod(&stm_flat)?;
        let stm_offs_dev: CudaSlice<i32> = stream.memcpy_stod(&stm_offs)?;
        let nstm_flat_dev: CudaSlice<i32> = stream.memcpy_stod(&nstm_flat)?;
        let nstm_offs_dev: CudaSlice<i32> = stream.memcpy_stod(&nstm_offs)?;

        // ── batch buffers (forward 内で alloc、 batch サイズ依存) ──
        let mut stm_acc = stream.alloc_zeros::<f32>(L1_SIZE * batch)?;
        let mut nstm_acc = stream.alloc_zeros::<f32>(L1_SIZE * batch)?;

        // bias 初期化 (batch broadcast)
        let bias_grid = ((L1_SIZE as u32).div_ceil(16), (batch as u32).div_ceil(16), 1);
        let bias_block = (16u32, 16u32, 1);
        unsafe {
            stream
                .launch_builder(&self.fn_add_bias_batch)
                .arg(&mut stm_acc)
                .arg(&self.weights.l0b)
                .arg(&(L1_SIZE as i32))
                .arg(&(batch as i32))
                .launch(LaunchConfig {
                    grid_dim: bias_grid,
                    block_dim: bias_block,
                    shared_mem_bytes: 0,
                })?;
            stream
                .launch_builder(&self.fn_add_bias_batch)
                .arg(&mut nstm_acc)
                .arg(&self.weights.l0b)
                .arg(&(L1_SIZE as i32))
                .arg(&(batch as i32))
                .launch(LaunchConfig {
                    grid_dim: bias_grid,
                    block_dim: bias_block,
                    shared_mem_bytes: 0,
                })?;
        }

        // FT sparse add (per-batch)
        let ft_grid = ((L1_SIZE as u32).div_ceil(256), batch as u32, 1);
        let ft_block = (256u32, 1, 1);
        unsafe {
            stream
                .launch_builder(&self.fn_ft_sparse_add_batch)
                .arg(&self.weights.l0w)
                .arg(&stm_flat_dev)
                .arg(&stm_offs_dev)
                .arg(&(L1_SIZE as i32))
                .arg(&(INPUT_DIM as i32))
                .arg(&(batch as i32))
                .arg(&mut stm_acc)
                .launch(LaunchConfig {
                    grid_dim: ft_grid,
                    block_dim: ft_block,
                    shared_mem_bytes: 0,
                })?;
            stream
                .launch_builder(&self.fn_ft_sparse_add_batch)
                .arg(&self.weights.l0w)
                .arg(&nstm_flat_dev)
                .arg(&nstm_offs_dev)
                .arg(&(L1_SIZE as i32))
                .arg(&(INPUT_DIM as i32))
                .arg(&(batch as i32))
                .arg(&mut nstm_acc)
                .launch(LaunchConfig {
                    grid_dim: ft_grid,
                    block_dim: ft_block,
                    shared_mem_bytes: 0,
                })?;
        }

        // SCReLU batch (elementwise)
        let n_total_ft = L1_SIZE * batch;
        let elem_grid_ft = ((n_total_ft as u32).div_ceil(256), 1, 1);
        let elem_block = (256u32, 1, 1);
        unsafe {
            stream
                .launch_builder(&self.fn_screlu_batch)
                .arg(&mut stm_acc)
                .arg(&(n_total_ft as i32))
                .launch(LaunchConfig {
                    grid_dim: elem_grid_ft,
                    block_dim: elem_block,
                    shared_mem_bytes: 0,
                })?;
            stream
                .launch_builder(&self.fn_screlu_batch)
                .arg(&mut nstm_acc)
                .arg(&(n_total_ft as i32))
                .launch(LaunchConfig {
                    grid_dim: elem_grid_ft,
                    block_dim: elem_block,
                    shared_mem_bytes: 0,
                })?;
        }

        // concat: (stm + nstm) col-major (2L1, batch)
        // 各 column b: 上半分 = stm_acc[b * L1..], 下半分 = nstm_acc[b * L1..]
        let mut concat = stream.alloc_zeros::<f32>(2 * L1_SIZE * batch)?;
        for b in 0..batch {
            let stm_src = stm_acc.slice(b * L1_SIZE..(b + 1) * L1_SIZE);
            let nstm_src = nstm_acc.slice(b * L1_SIZE..(b + 1) * L1_SIZE);
            stream.memcpy_dtod(
                &stm_src,
                &mut concat.slice_mut(b * 2 * L1_SIZE..b * 2 * L1_SIZE + L1_SIZE),
            )?;
            stream.memcpy_dtod(
                &nstm_src,
                &mut concat.slice_mut(b * 2 * L1_SIZE + L1_SIZE..(b + 1) * 2 * L1_SIZE),
            )?;
        }

        // ── value head (cuBLAS gemm) ──
        let mut h1 = stream.alloc_zeros::<f32>(L2_SIZE * batch)?;
        let h1_grid = ((L2_SIZE as u32).div_ceil(16), (batch as u32).div_ceil(16), 1);
        unsafe {
            stream
                .launch_builder(&self.fn_add_bias_batch)
                .arg(&mut h1)
                .arg(&self.weights.l1b)
                .arg(&(L2_SIZE as i32))
                .arg(&(batch as i32))
                .launch(LaunchConfig {
                    grid_dim: h1_grid,
                    block_dim: bias_block,
                    shared_mem_bytes: 0,
                })?;
        }
        gemm_with_bias_batch(
            blas,
            &self.weights.l1w,
            &concat,
            &mut h1,
            L2_SIZE,
            2 * L1_SIZE,
            batch,
        )?;
        let n_h1 = L2_SIZE * batch;
        unsafe {
            stream
                .launch_builder(&self.fn_leaky_relu_batch)
                .arg(&mut h1)
                .arg(&(n_h1 as i32))
                .launch(LaunchConfig {
                    grid_dim: ((n_h1 as u32).div_ceil(256), 1, 1),
                    block_dim: elem_block,
                    shared_mem_bytes: 0,
                })?;
        }

        let mut h2 = stream.alloc_zeros::<f32>(L3_SIZE * batch)?;
        let h2_grid = ((L3_SIZE as u32).div_ceil(16), (batch as u32).div_ceil(16), 1);
        unsafe {
            stream
                .launch_builder(&self.fn_add_bias_batch)
                .arg(&mut h2)
                .arg(&self.weights.l2b)
                .arg(&(L3_SIZE as i32))
                .arg(&(batch as i32))
                .launch(LaunchConfig {
                    grid_dim: h2_grid,
                    block_dim: bias_block,
                    shared_mem_bytes: 0,
                })?;
        }
        gemm_with_bias_batch(blas, &self.weights.l2w, &h1, &mut h2, L3_SIZE, L2_SIZE, batch)?;
        let n_h2 = L3_SIZE * batch;
        unsafe {
            stream
                .launch_builder(&self.fn_leaky_relu_batch)
                .arg(&mut h2)
                .arg(&(n_h2 as i32))
                .launch(LaunchConfig {
                    grid_dim: ((n_h2 as u32).div_ceil(256), 1, 1),
                    block_dim: elem_block,
                    shared_mem_bytes: 0,
                })?;
        }

        // value_out (1×L3 gemm)
        let mut value_out = stream.alloc_zeros::<f32>(batch)?;
        let vo_grid = ((1u32).div_ceil(16), (batch as u32).div_ceil(16), 1);
        unsafe {
            stream
                .launch_builder(&self.fn_add_bias_batch)
                .arg(&mut value_out)
                .arg(&self.weights.v_outb)
                .arg(&1i32)
                .arg(&(batch as i32))
                .launch(LaunchConfig {
                    grid_dim: vo_grid,
                    block_dim: bias_block,
                    shared_mem_bytes: 0,
                })?;
        }
        gemm_with_bias_batch(blas, &self.weights.v_outw, &h2, &mut value_out, 1, L3_SIZE, batch)?;

        // ── policy head SwiGLU ──
        let mut gate = stream.alloc_zeros::<f32>(ph * batch)?;
        let mut value_pre = stream.alloc_zeros::<f32>(ph * batch)?;
        let pol_grid = ((ph as u32).div_ceil(16), (batch as u32).div_ceil(16), 1);
        unsafe {
            stream
                .launch_builder(&self.fn_add_bias_batch)
                .arg(&mut gate)
                .arg(&self.weights.pol_w1b)
                .arg(&(ph as i32))
                .arg(&(batch as i32))
                .launch(LaunchConfig {
                    grid_dim: pol_grid,
                    block_dim: bias_block,
                    shared_mem_bytes: 0,
                })?;
            stream
                .launch_builder(&self.fn_add_bias_batch)
                .arg(&mut value_pre)
                .arg(&self.weights.pol_w3b)
                .arg(&(ph as i32))
                .arg(&(batch as i32))
                .launch(LaunchConfig {
                    grid_dim: pol_grid,
                    block_dim: bias_block,
                    shared_mem_bytes: 0,
                })?;
        }
        gemm_with_bias_batch(
            blas,
            &self.weights.pol_w1w,
            &concat,
            &mut gate,
            ph,
            2 * L1_SIZE,
            batch,
        )?;
        gemm_with_bias_batch(
            blas,
            &self.weights.pol_w3w,
            &concat,
            &mut value_pre,
            ph,
            2 * L1_SIZE,
            batch,
        )?;
        let n_ph = ph * batch;
        unsafe {
            stream
                .launch_builder(&self.fn_silu_batch)
                .arg(&mut gate)
                .arg(&(n_ph as i32))
                .launch(LaunchConfig {
                    grid_dim: ((n_ph as u32).div_ceil(256), 1, 1),
                    block_dim: elem_block,
                    shared_mem_bytes: 0,
                })?;
        }

        let mut swiglu = stream.alloc_zeros::<f32>(ph * batch)?;
        unsafe {
            stream
                .launch_builder(&self.fn_elem_mul_batch)
                .arg(&gate)
                .arg(&value_pre)
                .arg(&mut swiglu)
                .arg(&(n_ph as i32))
                .launch(LaunchConfig {
                    grid_dim: ((n_ph as u32).div_ceil(256), 1, 1),
                    block_dim: elem_block,
                    shared_mem_bytes: 0,
                })?;
        }

        let mut policy_logits = stream.alloc_zeros::<f32>(POLICY_DIMS * batch)?;
        let pl_grid = ((POLICY_DIMS as u32).div_ceil(16), (batch as u32).div_ceil(16), 1);
        unsafe {
            stream
                .launch_builder(&self.fn_add_bias_batch)
                .arg(&mut policy_logits)
                .arg(&self.weights.pol_ob)
                .arg(&(POLICY_DIMS as i32))
                .arg(&(batch as i32))
                .launch(LaunchConfig {
                    grid_dim: pl_grid,
                    block_dim: bias_block,
                    shared_mem_bytes: 0,
                })?;
        }
        gemm_with_bias_batch(
            blas,
            &self.weights.pol_ow,
            &swiglu,
            &mut policy_logits,
            POLICY_DIMS,
            ph,
            batch,
        )?;

        // ── CPU download ──
        let value_host = stream.memcpy_dtov(&value_out)?;
        let policy_host = stream.memcpy_dtov(&policy_logits)?;
        stream.synchronize()?;

        Ok((value_host, policy_host))
    }
}
