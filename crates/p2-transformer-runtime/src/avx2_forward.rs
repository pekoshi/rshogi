//! AVX2 path での Transformer forward (batch=1 専用、 fp32)。
//!
//! candle に依存せず、 重みは init 時に `[K, N]` (=gemm 用 pack 済み) に展開して
//! `AvxWeights` に格納する。 forward は scratch buffer を貫通し alloc-free。
//!
//! # PAD skip
//!
//! `MAX_TOKENS=128` のうち末尾 12 token は常に PAD。 forward 全体で `T_valid=116`
//! にして matmul / attention のサイズを削減する。
//!
//! # 安全性方針
//!
//! SIMD intrinsics と `target_feature` を伴うため、 関数全体が「AVX2 を要求する」
//! という契約のもとで unsafe を許容する。 個別の intrinsic call ごとに unsafe
//! ブロックを書かない方針 (詳細は `crates/p2-transformer-runtime/src/avx2/mod.rs`
//! のコメント)。

#![allow(unsafe_op_in_unsafe_fn)]

use anyhow::Result;
use candle_core::Tensor;
use std::arch::x86_64::*;

use p2_transformer::{MAX_TOKENS, POLICY_SIZE, TransformerTokens, VOCAB_SIZE};

use crate::avx2::{
    attention::mha_avx2,
    gelu::gelu_inplace_avx2,
    layer_norm::layer_norm_avx2,
    matmul::{gemm_a_bt, pack_weight_for_gemm},
};
use crate::model::{ModelHandle, RuntimeModelConfig};

/// PAD 末尾を除いた有効 token 数 (token slot 0..116 まで、 116..128 は PAD)。
pub const T_VALID: usize = 116;

/// 1 つの transformer block 分の重み (AVX2 用に pack 済み)。
struct BlockWeights {
    /// LN1
    ln1_gamma: Vec<f32>,
    ln1_beta: Vec<f32>,
    /// Q/K/V を結合: pack_weight `[K, 3N]` (K=d_model, N=d_model)、 結合された3つの packed weight
    qkv_packed: Vec<f32>, // [d_model, 3*d_model]
    /// Out proj weight + bias
    out_packed: Vec<f32>, // [d_model, d_model]
    out_bias: Vec<f32>,
    /// LN2
    ln2_gamma: Vec<f32>,
    ln2_beta: Vec<f32>,
    /// FFN
    fc1_packed: Vec<f32>, // [d_model, d_ff]
    fc1_bias: Vec<f32>,
    fc2_packed: Vec<f32>, // [d_ff, d_model]
    fc2_bias: Vec<f32>,
}

/// AVX2 用に整形された全 model 重み。
pub struct AvxWeights {
    pub cfg: RuntimeModelConfig,
    /// `[VOCAB_SIZE, d_model]` row-major
    tok_emb: Vec<f32>,
    /// `[MAX_TOKENS, d_model]` row-major
    pos_emb: Vec<f32>,
    blocks: Vec<BlockWeights>,
    ln_out_gamma: Vec<f32>,
    ln_out_beta: Vec<f32>,
    value_packed: Vec<f32>, // [d_model, 1]
    value_bias: Vec<f32>,
    policy_packed: Vec<f32>, // [d_model, POLICY_SIZE]
    policy_bias: Vec<f32>,
}

impl AvxWeights {
    /// candle `ModelHandle` から重みを展開する。 1 度だけ呼ぶ (init 時)。
    pub fn from_model(handle: &ModelHandle) -> Result<Self> {
        let cfg = handle.cfg.clone();
        let d_model = cfg.d_model;
        let d_ff = cfg.d_ff;
        let n_layers = cfg.n_layers;

        // ヘルパ: Tensor を flat Vec<f32> に dump
        fn to_vec(t: &Tensor) -> Result<Vec<f32>> {
            Ok(t.flatten_all()?.to_vec1::<f32>()?)
        }
        // Linear.weight は [out, in]
        fn pack_linear(t: &Tensor) -> Result<Vec<f32>> {
            let (out_d, in_d) = t.dims2()?;
            let src = to_vec(t)?;
            let mut dst = vec![0f32; in_d * out_d];
            pack_weight_for_gemm(&src, &mut dst, out_d, in_d);
            Ok(dst)
        }

        let tok_emb = to_vec(handle.tok_emb_tensor())?;
        let pos_emb = to_vec(handle.pos_emb_tensor())?;

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let blk = handle.block_tensors(i)?;
            // Q/K/V を [d_model, 3*d_model] に結合 pack
            // 各 linear weight は [d_model, d_model] = [out, in]
            let q_packed = pack_linear(blk.q_weight)?;
            let k_packed = pack_linear(blk.k_weight)?;
            let v_packed = pack_linear(blk.v_weight)?;
            // qkv_packed: [d_model (=K), 3*d_model (=N)] row-major
            //   = [K, N_q | K, N_k | K, N_v]
            let mut qkv = vec![0f32; d_model * 3 * d_model];
            for kk in 0..d_model {
                for nn in 0..d_model {
                    qkv[kk * 3 * d_model + nn] = q_packed[kk * d_model + nn];
                    qkv[kk * 3 * d_model + d_model + nn] = k_packed[kk * d_model + nn];
                    qkv[kk * 3 * d_model + 2 * d_model + nn] = v_packed[kk * d_model + nn];
                }
            }
            let out_packed = pack_linear(blk.out_weight)?;
            let out_bias = to_vec(blk.out_bias)?;
            let fc1_packed = pack_linear(blk.fc1_weight)?;
            let fc1_bias = to_vec(blk.fc1_bias)?;
            let fc2_packed = pack_linear(blk.fc2_weight)?;
            let fc2_bias = to_vec(blk.fc2_bias)?;
            blocks.push(BlockWeights {
                ln1_gamma: to_vec(blk.ln1_gamma)?,
                ln1_beta: to_vec(blk.ln1_beta)?,
                qkv_packed: qkv,
                out_packed,
                out_bias,
                ln2_gamma: to_vec(blk.ln2_gamma)?,
                ln2_beta: to_vec(blk.ln2_beta)?,
                fc1_packed,
                fc1_bias,
                fc2_packed,
                fc2_bias,
            });
            let _ = d_ff; // d_ff is implicit via fc1 dims
        }

        let ln_out = handle.ln_out_tensors()?;
        let value = handle.value_head_tensors()?;
        let policy = handle.policy_head_tensors()?;

        Ok(Self {
            cfg,
            tok_emb,
            pos_emb,
            blocks,
            ln_out_gamma: to_vec(ln_out.0)?,
            ln_out_beta: to_vec(ln_out.1)?,
            value_packed: pack_linear(value.0)?,
            value_bias: to_vec(value.1)?,
            policy_packed: pack_linear(policy.0)?,
            policy_bias: to_vec(policy.1)?,
        })
    }
}

/// AVX2 forward 用の scratch buffer (heap allocation hot path 排除)。
pub struct AvxScratch {
    /// `[T_VALID, d_model]` activations (x、 layer 入力)
    x: Vec<f32>,
    /// `[T_VALID, 3*d_model]` Q/K/V projection 出力
    qkv: Vec<f32>,
    /// `[T_VALID, d_model]` attention 出力 (out_proj 前)
    attn_out: Vec<f32>,
    /// `[T_VALID, d_model]` attn out_proj 後 (residual 加算用)
    out_proj: Vec<f32>,
    /// `[T_VALID, d_ff]` FFN 中間
    ff_mid: Vec<f32>,
    /// `[n_heads * T_VALID * T_VALID]` attention 行列
    attn_scratch: Vec<f32>,
}

impl AvxScratch {
    pub fn new(cfg: &RuntimeModelConfig) -> Self {
        let d_model = cfg.d_model;
        let d_ff = cfg.d_ff;
        let n_heads = cfg.n_heads;
        Self {
            x: vec![0f32; T_VALID * d_model],
            qkv: vec![0f32; T_VALID * 3 * d_model],
            attn_out: vec![0f32; T_VALID * d_model],
            out_proj: vec![0f32; T_VALID * d_model],
            ff_mid: vec![0f32; T_VALID * d_ff],
            attn_scratch: vec![0f32; n_heads * T_VALID * T_VALID],
        }
    }
}

/// AVX2 forward (batch=1)。 `value_out`: 1 要素、 `policy_out`: POLICY_SIZE 要素を書き込む。
///
/// # Safety
/// AVX2 + FMA を要求。 呼び出し元で `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`
/// を保証する責務がある。
#[target_feature(enable = "avx2,fma")]
pub unsafe fn forward_batch1_avx2(
    weights: &AvxWeights,
    scratch: &mut AvxScratch,
    tokens: &TransformerTokens,
    value_out: &mut f32,
    policy_out: &mut [f32],
) {
    debug_assert_eq!(policy_out.len(), POLICY_SIZE);
    let cfg = &weights.cfg;
    let d_model = cfg.d_model;
    let d_ff = cfg.d_ff;
    let n_heads = cfg.n_heads;
    let d_head = d_model / n_heads;
    let eps = cfg.ln_eps as f32;

    // SAFETY: 全 SIMD 呼び出しは target_feature(avx2,fma) で有効。
    // scratch / weights のサイズは init 時に整合済み (assert)。

    // 1. token embedding + position embedding
    //    x[t, :] = tok_emb[tokens[t], :] + pos_emb[t, :]   (t < T_VALID)
    for t in 0..T_VALID {
        let tok_id = tokens.tokens[t] as usize;
        debug_assert!(tok_id < VOCAB_SIZE);
        let tok_row = &weights.tok_emb[tok_id * d_model..(tok_id + 1) * d_model];
        let pos_row = &weights.pos_emb[t * d_model..(t + 1) * d_model];
        let x_row = &mut scratch.x[t * d_model..(t + 1) * d_model];
        let mut i = 0usize;
        while i + 8 <= d_model {
            let a = _mm256_loadu_ps(tok_row.as_ptr().add(i));
            let b = _mm256_loadu_ps(pos_row.as_ptr().add(i));
            _mm256_storeu_ps(x_row.as_mut_ptr().add(i), _mm256_add_ps(a, b));
            i += 8;
        }
        while i < d_model {
            x_row[i] = tok_row[i] + pos_row[i];
            i += 1;
        }
    }

    // 2. for each block
    for blk in weights.blocks.iter() {
        // 2.1 LN1 (in-place on a scratch copy)
        // residual 用に x を残しつつ LN1 結果を別 buffer に置く → 同 layout の attn_out を流用
        scratch.attn_out.copy_from_slice(&scratch.x);
        layer_norm_avx2(
            &mut scratch.attn_out,
            &blk.ln1_gamma,
            &blk.ln1_beta,
            T_VALID,
            d_model,
            eps,
        );

        // 2.2 Q/K/V projection (fused: [T_VALID, d_model] @ [d_model, 3*d_model] → [T_VALID, 3*d_model])
        gemm_a_bt(
            &scratch.attn_out, // input
            &blk.qkv_packed,   // packed weight (already [K, N=3*d_model] row-major)
            &mut scratch.qkv,
            None,
            T_VALID,
            3 * d_model,
            d_model,
        );

        // qkv split: Q, K, V は qkv の N 軸 0..d, d..2d, 2d..3d
        // attention は Q/K/V を分離 buffer で受けるが、 stride 読みなら 1 つの buffer のままでも
        // OK。 mha_avx2 は q/k/v slice を要求するので、 view を作る。
        // [T_VALID, 3*d_model] から Q [T_VALID, d_model] (stride 3*d_model) を取り出すには
        // 都度 copy するか、 mha_avx2 を stride 引数対応にする。 ここでは簡潔さ優先で copy する。
        // (T_VALID * d_model = 116 * 192 = 22272 f32 = 87 KB / バッファ、 L2 内、 cost 小)
        let three_d = 3 * d_model;
        // reuse: q→attn_out, k→qkv first half (不可: qkv は元 layout)、 v→out_proj 等
        // すっきりさせるため、 q / k / v それぞれ用 buffer を持つ。
        // → scratch に追加するのは無駄、 既存 buffer を使い回す:
        //   q → scratch.attn_out (LN1 結果を上書き、 後で不要)
        //   k → 同 qkv buffer の k 領域を 1 本に compact、 これは新 buffer 要
        //   v → 同上
        // 簡略化: out_proj と ff_mid (d_ff>=d_model) を temp として使う
        let q_view = scratch.attn_out.as_mut_ptr();
        let k_view = scratch.out_proj.as_mut_ptr();
        let v_view = scratch.ff_mid.as_mut_ptr();
        // SAFETY: target_feature(avx2,fma) 内で実行、 アドレス範囲は事前 alloc 済み
        // (scratch.qkv: T_VALID * 3 * d_model、 q/k/v_view: T_VALID * d_model)。
        debug_assert_eq!(d_model % 8, 0);
        for t in 0..T_VALID {
            let src = scratch.qkv.as_ptr().add(t * three_d);
            let qdst = q_view.add(t * d_model);
            let kdst = k_view.add(t * d_model);
            let vdst = v_view.add(t * d_model);
            let mut d = 0usize;
            while d < d_model {
                _mm256_storeu_ps(qdst.add(d), _mm256_loadu_ps(src.add(d)));
                _mm256_storeu_ps(kdst.add(d), _mm256_loadu_ps(src.add(d_model + d)));
                _mm256_storeu_ps(vdst.add(d), _mm256_loadu_ps(src.add(2 * d_model + d)));
                d += 8;
            }
        }
        let q_slice = std::slice::from_raw_parts(q_view, T_VALID * d_model);
        let k_slice = std::slice::from_raw_parts(k_view, T_VALID * d_model);
        let v_slice = std::slice::from_raw_parts(v_view, T_VALID * d_model);

        // 2.3 attention (in-place qkv buffer reuse for output)
        // 出力は scratch.qkv[0..T_VALID*d_model] (前半に上書き) を使う
        let attn_out_slice =
            std::slice::from_raw_parts_mut(scratch.qkv.as_mut_ptr(), T_VALID * d_model);
        mha_avx2(
            q_slice,
            k_slice,
            v_slice,
            attn_out_slice,
            &mut scratch.attn_scratch,
            T_VALID,
            T_VALID,
            n_heads,
            d_head,
        );

        // 2.4 out_proj: attn_out @ W_out + b_out
        gemm_a_bt(
            attn_out_slice,
            &blk.out_packed,
            &mut scratch.out_proj,
            Some(&blk.out_bias),
            T_VALID,
            d_model,
            d_model,
        );

        // 2.5 residual: x += out_proj
        {
            let xp = scratch.x.as_mut_ptr();
            let op = scratch.out_proj.as_ptr();
            let n = T_VALID * d_model;
            let n_aligned = n & !7;
            let mut i = 0usize;
            while i < n_aligned {
                let a = _mm256_loadu_ps(xp.add(i));
                let b = _mm256_loadu_ps(op.add(i));
                _mm256_storeu_ps(xp.add(i), _mm256_add_ps(a, b));
                i += 8;
            }
            while i < n {
                *xp.add(i) += *op.add(i);
                i += 1;
            }
        }

        // 2.6 LN2 → attn_out buffer 再利用
        scratch.attn_out.copy_from_slice(&scratch.x);
        layer_norm_avx2(
            &mut scratch.attn_out,
            &blk.ln2_gamma,
            &blk.ln2_beta,
            T_VALID,
            d_model,
            eps,
        );

        // 2.7 FFN: fc1 → GELU → fc2
        gemm_a_bt(
            &scratch.attn_out,
            &blk.fc1_packed,
            &mut scratch.ff_mid,
            Some(&blk.fc1_bias),
            T_VALID,
            d_ff,
            d_model,
        );
        gelu_inplace_avx2(&mut scratch.ff_mid[..T_VALID * d_ff]);
        gemm_a_bt(
            &scratch.ff_mid,
            &blk.fc2_packed,
            &mut scratch.out_proj,
            Some(&blk.fc2_bias),
            T_VALID,
            d_model,
            d_ff,
        );

        // 2.8 residual: x += out_proj
        {
            let xp = scratch.x.as_mut_ptr();
            let op = scratch.out_proj.as_ptr();
            let n = T_VALID * d_model;
            let n_aligned = n & !7;
            let mut i = 0usize;
            while i < n_aligned {
                let a = _mm256_loadu_ps(xp.add(i));
                let b = _mm256_loadu_ps(op.add(i));
                _mm256_storeu_ps(xp.add(i), _mm256_add_ps(a, b));
                i += 8;
            }
            while i < n {
                *xp.add(i) += *op.add(i);
                i += 1;
            }
        }
    }

    // 3. final LN → CLS slice → value / policy heads
    layer_norm_avx2(
        &mut scratch.x,
        &weights.ln_out_gamma,
        &weights.ln_out_beta,
        T_VALID,
        d_model,
        eps,
    );

    // CLS = row 0
    let cls_slice = &scratch.x[..d_model];

    // value: [1, d_model] @ [d_model, 1] + bias[0]
    let mut v_out = [0f32; 1];
    gemm_a_bt(
        cls_slice,
        &weights.value_packed,
        &mut v_out,
        Some(&weights.value_bias),
        1,
        1,
        d_model,
    );
    *value_out = v_out[0];

    // policy: [1, d_model] @ [d_model, POLICY_SIZE] + bias
    gemm_a_bt(
        cls_slice,
        &weights.policy_packed,
        policy_out,
        Some(&weights.policy_bias),
        1,
        POLICY_SIZE,
        d_model,
    );

    // 防止: MAX_TOKENS の未使用 token (T_VALID..MAX_TOKENS) は無視する設計。
    let _ = MAX_TOKENS;
}
