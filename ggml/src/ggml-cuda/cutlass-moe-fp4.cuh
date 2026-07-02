// CUTLASS grouped NVFP4 MoE bridge for ggml_cuda_mul_mat_id.
// Replaces the per-expert ggml_cuda_mul_mat storm with ONE grouped block-scaled
// FP4 GEMM (sm120) producing bf16, rescaled to f32. Numerics proven by
// .frankencoder/engine/fp4moe-port/validate_glue.cu (relFrobErr 1.66e-3).
//
// This header is pure C ABI (no CUTLASS/ggml types) so ggml-cuda.cu can call it
// regardless of whether the CUTLASS TU is built.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// One grouped NVFP4 MoE GEMM.
//   w_blocks          : ggml NVFP4 expert weights (block_nvfp4), logical [K, N, E];
//                       byte offset of expert e / row n / K-block b = e*nb02 + n*nb01 + b*36.
//   src1_sorted       : [Mtot, K] f32 activations, rows grouped by expert (Mtot = sum tokens).
//   dst_sorted        : [Mtot, N] f32 output (row-major), written in-place.
//   tokens_per_expert : host int32[E], rows assigned to each expert (0 allowed).
//   E,N,K             : expert count, output dim, contraction dim (N,K multiples of 128/64).
//   nb01,nb02         : ggml weight row / expert byte strides.
// Returns 0 on success, non-zero on failure (caller should fall back to per-expert path).
int ggml_cuda_cutlass_moe_nvfp4(
    const void * w_blocks, const float * src1_sorted, float * dst_sorted,
    const int32_t * tokens_per_expert, int E, int N, int K,
    size_t nb01, size_t nb02, cudaStream_t stream);

// True if the current build has the CUTLASS grouped FP4 path compiled in and the
// given shape is supported (N,K multiples of 64, sm120+). Cheap; no allocation.
int ggml_cuda_cutlass_moe_nvfp4_supported(int E, int N, int K);

// ZERO-SYNC prefill path (2026-07-02). Fully stream-ordered: no host syncs, no
// per-call cudaMalloc (persistent grow-only scratch), no src1 materialization.
// Hooks the mmq.cu mul_mat_id path using its device-side routing arrays:
//   src1          : UNSORTED f32 activations (gathered internally via ids_src1)
//   ids_src1      : [Mtot] compact row -> src1 column index (from mm_ids_helper)
//   ids_dst       : [Mtot] compact row -> dst column index
//   expert_bounds : [E+1] compact-row bounds per expert
//   dst           : f32 output, written as dst[ids_dst[m]*s1 + n]
//   Mtot          : total compact rows (ne12 * n_expert_used)
//   s11 / s1      : src1 / dst column strides (elements)
// Returns 0 on success; non-zero -> caller must run the MMQ path instead.
// Routing (mm_ids_helper) now runs INSIDE the bridges and is cached across the
// gate/up/down calls of one MoE layer (same ids tensor, <=3 uses before forced
// recompute so allocator buffer-aliasing can never serve stale routing). The
// gathered+quantized A operand is additionally reused across gate->up (same
// src1/K). Callers pass the raw ids tensor data + its strides.
struct ggml_cuda_moe_ids_args {
    const int32_t * ids_data;   // ids tensor (device)
    int n_experts;              // ne02
    int n_tokens;               // ne12
    int n_expert_used;          // ids->ne[0]
    int nchannels_y;            // ne11
    int si1;                    // ids row stride (elements)
    int sis1;                   // src1 sample stride ratio (nb12/nb11)
};

// fuse_w (optional, may be NULL): routing weights, contiguous [1, n_expert_used,
// n_tokens] f32 — flat index it*n_expert_used+iex == ids_dst[m]. When set, the
// graph-level ggml_mul(experts, weights) is fused into the scatter epilogue and
// dst is the MUL node's output tensor.
// accum_neu > 0: additionally fuse the sum-over-experts — dst is the FINAL
// [N, n_tokens] MoE output (pre-zeroed here), rows accumulated per token.
int ggml_cuda_cutlass_moe_nvfp4_prefill(
    const void * w_blocks, const float * src1, const ggml_cuda_moe_ids_args * ids,
    float * dst, int E, int N, int K, int Mtot, int64_t s11, int64_t s1,
    size_t nb01, size_t nb02, const float * fuse_w, int accum_neu, cudaStream_t stream);

// Same zero-sync bridge for GGML_TYPE_F16 experts (cutlass-moe-f16.cu):
// ONE CUTLASS grouped f16 GEMM (device-side problem sizes) instead of the
// per-expert cuBLAS storm. Weights used in place (no repack); w_f16 points
// at the expert tensor, strides nb01/nb02 in bytes. Returns 0 on success.
int ggml_cuda_cutlass_moe_f16_prefill(
    const void * w_f16, const float * src1, const ggml_cuda_moe_ids_args * ids,
    float * dst, int E, int N, int K, int Mtot, int64_t s11, int64_t s1,
    size_t nb01, size_t nb02, const float * fuse_w, int accum_neu, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
