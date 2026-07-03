// Glue kernels bridging ggml's NVFP4 block layout <-> CUTLASS grouped-GEMM
// operand/scale layouts. Pure-CUDA C ABI (no CUTLASS or ggml headers) so it
// always compiles and links regardless of GGML_CUDA_CUTLASS_FP4.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Repack ggml block_nvfp4 weights -> CUTLASS contiguous e2m1 + swizzled SFB.
//   src0_blocks : ggml weight tensor data (block_nvfp4), expert e / row n at
//                 byte offset e*nb02 + n*nb01, then K/64 contiguous blocks.
//   b_e2m1      : out [E, N, K/2] u8 (consecutive-pair e2m1 packing)
//   sfb         : out swizzled ue4m3 scales, per-expert stride N*(K/16) bytes
// Run ONCE per weight tensor at first use (weights are static); cache the result.
void ggml_cuda_nvfp4_repack_weights(
    const void * src0_blocks, void * b_e2m1, void * sfb,
    int E, int N, int K, size_t nb01, size_t nb02, cudaStream_t stream);

// Two-level NVFP4 scaling (matches vLLM): compute a per-row global fp32 scale
// g[m] = row_amax / 2688 so that block (e4m3) scales never underflow to zero on
// small-magnitude activation rows (e.g. post-SwiGLU down-proj input). Activations
// are quantized in the a/g domain; the GEMM output row m is later multiplied by g[m].
//   src1_sorted : [M, K] f32 row-major
//   row_scale   : out [M] f32, g[m] (1.0 for all-zero rows)
void ggml_cuda_nvfp4_act_row_scale(
    const float * src1_sorted, float * row_scale, int M, int K, cudaStream_t stream);

// Quantize f32 activations (rows already grouped by expert) -> e2m1 + swizzled SFA.
//   src1_sorted   : [M, K] f32 row-major (M = total sorted rows)
//   row_scale     : [M] f32 per-row global scale g[m] (from act_row_scale)
//   a_e2m1        : out [M, K/2] u8
//   sfa           : out swizzled ue4m3, per-expert base sf_offsets[e]*(K/16)
//   row_expert    : [M] int32, expert id for each sorted row
//   expert_offsets: [E] int32, prefix sum of M_e (row base per expert)
//   sf_offsets    : [E] int32, prefix sum of round_up(M_e,128)
void ggml_cuda_nvfp4_quant_acts(
    const float * src1_sorted, const float * row_scale, void * a_e2m1, void * sfa,
    const int32_t * row_expert, const int32_t * expert_offsets,
    const int32_t * sf_offsets, int M, int K, cudaStream_t stream);

// bfloat16 -> f32 elementwise (GEMM output is bf16 on SM120).
void ggml_cuda_bf16_to_f32_buf(const void * src_bf16, float * dst,
                               int64_t n, cudaStream_t stream);

// ---- zero-sync prefill path (2026-07-02) ----
// sf_offsets[e] = prefix sum of round_up(Me,128) computed ON DEVICE from
// expert_bounds (no host sync). sf_offsets has E entries.
void ggml_cuda_nvfp4_sf_offsets(const int32_t * expert_bounds, int32_t * sf_offsets,
                                int E, cudaStream_t stream);

// Fused gather + per-row global scale + NVFP4 quantize, one pass, reading the
// UNSORTED src1 directly through ids_src1 (the mm_ids_helper gather map).
// Replaces get_rows + act_row_scale + quant_acts (3 kernels, 2 extra passes).
//   src1        : f32 activations, column i at byte offset i*s11*4
//   ids_src1    : [Mtot] device gather map (compact row m -> src1 column)
//   expert_bounds: [E+1] device compact-row bounds per expert
//   sf_offsets  : [E] device (from ggml_cuda_nvfp4_sf_offsets)
//   row_scale   : out [Mtot] f32 g[m]
//   a_e2m1      : out [Mtot, K/2] u8
//   sfa         : out swizzled ue4m3 (per-expert base sf_offsets[e]*(K/16))
// K must be <= 12288 (row cached in dynamic shared memory).
void ggml_cuda_nvfp4_gather_quant(
    const float * src1, const int32_t * ids_src1, const int32_t * expert_bounds,
    const int32_t * sf_offsets, float * row_scale, void * a_e2m1, void * sfa,
    int E, int Mtot, int K, int64_t s11, cudaStream_t stream);

// Fused scatter + rescale + bf16->f32: dst[ids_dst[m]*s1 + n] = bf16(D[m,n]) * g[m].
// fuse_w (optional, may be NULL): routing weights indexed by ids_dst[m] — fuses
// the graph-level ggml_mul(experts, weights) into the scatter (dst *= fuse_w).
// accum_neu > 0: rows are accumulated per TOKEN (dst row = ids_dst[m]/accum_neu,
// atomicAdd) — fuses the sum-over-experts too. dst must be pre-zeroed.
void ggml_cuda_nvfp4_scatter_rowscaled(
    const void * d_bf16, const float * row_scale, const int32_t * ids_dst,
    float * dst, int Mtot, int N, int64_t s1, const float * fuse_w, int accum_neu,
    cudaStream_t stream);

// bf16 -> f32 with per-row rescale: dst[m,n] = bf16(src[m,n]) * row_scale[m].
// Undoes the per-row global activation scale folded into the GEMM inputs.
void ggml_cuda_bf16_to_f32_rowscaled(const void * src_bf16, float * dst,
                                     const float * row_scale, int M, int N,
                                     cudaStream_t stream);

// ---- F16 MoE prefill glue (2026-07-02) ----
// gather + f32->f16 convert through ids_src1 (one block/row, vectorized)
void ggml_cuda_moe_gather_f32_to_f16(
    const float * src1, const int32_t * ids_src1, void * dst_f16,
    int Mtot, int K, int64_t s11, cudaStream_t stream);

// scatter f32 rows through ids_dst (one block/row, vectorized)
void ggml_cuda_moe_scatter_f32(
    const float * src, const int32_t * ids_dst, float * dst,
    int Mtot, int N, int64_t s1, const float * fuse_w, int accum_neu, cudaStream_t stream);
// ---- no-atomic expert-sum (2026-07-02): inverse-permutation gather ----
// inv[ids_dst[m]] = m (ids_dst must be a bijection over [0, Mtot))
void ggml_cuda_moe_invert_ids(const int32_t * ids_dst, int32_t * inv, int Mtot, cudaStream_t stream);
// One block/token: sums the token's neu compact rows in registers, writes dst
// once (no atomics, no pre-zero). Replaces scatter_rowscaled<accum=true>.
// neu must be <= 16.
void ggml_cuda_nvfp4_gather_accum_rowscaled(
    const void * d_bf16, const float * row_scale, const int32_t * inv,
    float * dst, int n_tokens, int N, int64_t s1, const float * fuse_w, int neu,
    cudaStream_t stream);
// f32-src variant (F16 MoE path)
void ggml_cuda_moe_gather_accum_f32(
    const float * src, const int32_t * inv, float * dst,
    int n_tokens, int N, int64_t s1, const float * fuse_w, int neu, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
