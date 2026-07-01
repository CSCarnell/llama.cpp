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

#ifdef __cplusplus
}
#endif
