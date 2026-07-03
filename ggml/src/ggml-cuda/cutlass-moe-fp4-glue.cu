// Glue kernels: ggml block_nvfp4 <-> CUTLASS NVFP4 operand/scale layouts.
// Pure CUDA, no CUTLASS/ggml headers. See cutlass-moe-fp4-glue.cuh.
//
// Scale convention: ggml stores ue4m3 scale bytes whose STANDARD ue4m3 value
// equals (amax/6); its e2m1 LUT {0,1,2,3,4,6,8,12} is 2x the standard e2m1 and
// its decode multiplies by 0.5, so the two factors cancel and ggml's effective
// dequant is e2m1_std[code]*ue4m3_std(scale) — identical to CUTLASS with
// alpha=1.0. Hence weight scale bytes copy through unchanged and the 4-bit
// codes are bit-compatible with cutlass::float_e2m1_t.

#include "cutlass-moe-fp4-glue.cuh"

#include <cuda_fp16.h>

#define QK_NVFP4      64
#define QK_NVFP4_SUB  16
#define NVFP4_NSUB    (QK_NVFP4 / QK_NVFP4_SUB)   // 4
#define NVFP4_BYTES   36                          // 4 scale (d[4]) + 32 qs = 36 (matches ggml block_nvfp4)

// ----------------------------------------------------------------------------
// device helpers
// ----------------------------------------------------------------------------

// ggml_fp32_to_ue4m3 ported verbatim (standard ue4m3, bias 7).
__device__ __forceinline__ uint8_t d_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) return 0;
    if (x > 448.0f) x = 448.0f;
    uint32_t bits;
    memcpy(&bits, &x, 4);
    int fp32_exp = ((bits >> 23) & 0xFF) - 127;
    int fp32_man = (bits >> 20) & 0x7;
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {
        int man = (int)(x * 512.0f + 0.5f);
        if (man > 7) man = 7;
        if (man < 1) return 0;
        return (uint8_t)man;
    }
    if (ue4m3_exp > 15) return 0x7E;                 // true overflow -> max finite 448
    int round_bit = (bits >> 19) & 1;
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) { ue4m3_man = 0; ue4m3_exp++; if (ue4m3_exp > 15) return 0x7E; }
    if (ue4m3_exp == 15 && ue4m3_man > 6) ue4m3_man = 6;  // 0x7F is NaN; clamp to 448
    return (uint8_t)((ue4m3_exp << 3) | ue4m3_man);
}

// standard ue4m3 -> f32 (no 0.5 factor; matches cutlass::float_ue4m3_t)
__device__ __forceinline__ float d_ue4m3_to_fp32_std(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int exp = (x >> 3) & 0xF;
    int man = x & 0x7;
    if (exp == 0) return ldexpf((float)man, -9);
    return ldexpf(1.0f + (float)man / 8.0f, exp - 7);
}

// nearest standard-e2m1 4-bit code for value x given block scale.
__device__ __forceinline__ uint8_t d_e2m1_code(float x, float scale) {
    const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float ax = fabsf(x);
    float inv = scale > 0.f ? ax / scale : 0.f;  // compare in scaled domain
    int best = 0;
    float berr = fabsf(lut[0] - inv);
    for (int i = 1; i < 8; i++) {
        float e = fabsf(lut[i] - inv);
        if (e < berr) { berr = e; best = i; }
    }
    uint8_t code = (uint8_t)best;
    if (x < 0.f && best != 0) code |= 0x8;
    return code;
}

// byte offset of scale (mIdx,kIdx) within a swizzled SF block: [mTiles,kTiles,32,4,4]
__device__ __forceinline__ int64_t d_sf_swizzle(int mIdx, int kIdx, int numKTiles) {
    int mTile  = mIdx >> 7;
    int outerM = mIdx & 31;
    int innerM = (mIdx >> 5) & 3;
    int kTile  = kIdx >> 2;
    int innerK = kIdx & 3;
    return (((int64_t)(mTile * numKTiles + kTile)) << 9)
           | (int64_t)((outerM << 4) | (innerM << 2) | innerK);
}

// ----------------------------------------------------------------------------
// weight repack: one thread per (weight row, 16-element sub-block)
// ----------------------------------------------------------------------------
__global__ void k_repack_weights(
        const uint8_t * src0, uint8_t * b_e2m1, uint8_t * sfb,
        int E, int N, int K, size_t nb01, size_t nb02, int numKTiles) {
    const int nsubK = K / QK_NVFP4_SUB;       // sub-blocks (=scales) per row
    const long total = (long)E * N * nsubK;
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int g  = (int)(tid % nsubK);        // global sub-block along K
    const long rn = tid / nsubK;              // global weight row (e*N + n)
    const int n  = (int)(rn % N);
    const int e  = (int)(rn / N);

    const int blk = g / NVFP4_NSUB;           // 64-elem block index
    const int s   = g % NVFP4_NSUB;           // sub-block within block
    const uint8_t * blkptr =
        src0 + (size_t)e * nb02 + (size_t)n * nb01 + (size_t)blk * NVFP4_BYTES;
    const uint8_t * d  = blkptr;              // 4 scale bytes
    const uint8_t * qs = blkptr + NVFP4_NSUB; // 32 qs bytes (offset 4)

    // scale: copy ggml ue4m3 byte straight through, into swizzled SFB position
    uint8_t * sfb_e = sfb + (size_t)e * N * (K / QK_NVFP4_SUB);
    sfb_e[d_sf_swizzle(n, g, numKTiles)] = d[s];

    // e2m1: rearrange ggml's split nibbles (pos t<8 low of qs[s*8+t];
    // t>=8 high of qs[s*8+t-8]) into consecutive-pair output bytes.
    uint8_t * out_row = b_e2m1 + ((size_t)e * N + n) * (K / 2);
    const int out_base = g * (QK_NVFP4_SUB / 2);    // 8 bytes per sub-block
    for (int b = 0; b < QK_NVFP4_SUB / 2; b++) {
        int t0 = 2 * b, t1 = 2 * b + 1;
        uint8_t c0 = (t0 < 8) ? (qs[s * 8 + t0] & 0xF) : (qs[s * 8 + (t0 - 8)] >> 4);
        uint8_t c1 = (t1 < 8) ? (qs[s * 8 + t1] & 0xF) : (qs[s * 8 + (t1 - 8)] >> 4);
        out_row[out_base + b] = c0 | (c1 << 4);
    }
}

// ----------------------------------------------------------------------------
// per-row global scale g[m] = row_amax / 2688  (2688 = 6 * 448 = e2m1_max * e4m3_max)
// one CUDA block per row; block-stride reduction over K with shared-mem reduce.
// ----------------------------------------------------------------------------
__global__ void k_act_row_scale(const float * src, float * row_scale, int M, int K) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const float * row = src + (size_t)m * K;
    float amax = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) { float a = fabsf(row[k]); amax = a > amax ? a : amax; }
    __shared__ float s[256];
    s[threadIdx.x] = amax;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + off]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float a = s[0];
        row_scale[m] = a > 0.f ? a / 2688.0f : 1.0f;
    }
}

// ----------------------------------------------------------------------------
// activation quant: one thread per (sorted row, 16-element sub-block)
// Quantizes in the a/g domain (g = row_scale[m]) so block scales never underflow.
// ----------------------------------------------------------------------------
__global__ void k_quant_acts(
        const float * src, const float * row_scale, uint8_t * a_e2m1, uint8_t * sfa,
        const int32_t * row_expert, const int32_t * expert_offsets,
        const int32_t * sf_offsets, int M, int K, int numKTiles) {
    const int nsubK = K / QK_NVFP4_SUB;
    const long total = (long)M * nsubK;
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int g = (int)(tid % nsubK);     // sub-block along K
    const int m = (int)(tid / nsubK);     // sorted row
    const float invg = 1.0f / row_scale[m];                       // 1 / g[m]
    const float * xb = src + (size_t)m * K + (size_t)g * QK_NVFP4_SUB;

    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < QK_NVFP4_SUB; j++) { float a = fabsf(xb[j] * invg); amax = a > amax ? a : amax; }

    uint8_t ue = d_fp32_to_ue4m3(amax / 6.0f);
    float scale = d_ue4m3_to_fp32_std(ue);

    // SFA position: per-expert padded-row base + swizzle(local row, g)
    const int e   = row_expert[m];
    const int r   = m - expert_offsets[e];        // local row within expert
    uint8_t * sfa_e = sfa + (size_t)sf_offsets[e] * (K / QK_NVFP4_SUB);
    sfa_e[d_sf_swizzle(r, g, numKTiles)] = ue;

    uint8_t * out_row = a_e2m1 + (size_t)m * (K / 2);
    const int out_base = g * (QK_NVFP4_SUB / 2);
    for (int b = 0; b < QK_NVFP4_SUB / 2; b++) {
        uint8_t c0 = d_e2m1_code(xb[2 * b]     * invg, scale);
        uint8_t c1 = d_e2m1_code(xb[2 * b + 1] * invg, scale);
        out_row[out_base + b] = c0 | (c1 << 4);
    }
}

// ----------------------------------------------------------------------------
// bf16 -> f32
// ----------------------------------------------------------------------------
__global__ void k_bf16_to_f32(const uint16_t * src, float * dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t bits = (uint32_t)src[i] << 16;
    float f; memcpy(&f, &bits, 4);
    dst[i] = f;
}

// bf16 -> f32 with per-row rescale: dst[m*N+n] = bf16(src) * row_scale[m]
__global__ void k_bf16_to_f32_rowscaled(
        const uint16_t * src, float * dst, const float * row_scale, int M, int N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)M * N) return;
    uint32_t bits = (uint32_t)src[i] << 16;
    float f; memcpy(&f, &bits, 4);
    dst[i] = f * row_scale[i / N];
}

// ----------------------------------------------------------------------------
// launchers
// ----------------------------------------------------------------------------
extern "C" void ggml_cuda_nvfp4_repack_weights(
        const void * src0_blocks, void * b_e2m1, void * sfb,
        int E, int N, int K, size_t nb01, size_t nb02, cudaStream_t stream) {
    const int numKTiles = (K + 63) / 64;
    const long total = (long)E * N * (K / QK_NVFP4_SUB);
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    k_repack_weights<<<(unsigned)blocks, threads, 0, stream>>>(
        (const uint8_t *)src0_blocks, (uint8_t *)b_e2m1, (uint8_t *)sfb,
        E, N, K, nb01, nb02, numKTiles);
}

extern "C" void ggml_cuda_nvfp4_act_row_scale(
        const float * src1_sorted, float * row_scale, int M, int K, cudaStream_t stream) {
    k_act_row_scale<<<(unsigned)M, 256, 0, stream>>>(src1_sorted, row_scale, M, K);
}

extern "C" void ggml_cuda_nvfp4_quant_acts(
        const float * src1_sorted, const float * row_scale, void * a_e2m1, void * sfa,
        const int32_t * row_expert, const int32_t * expert_offsets,
        const int32_t * sf_offsets, int M, int K, cudaStream_t stream) {
    const int numKTiles = (K + 63) / 64;
    const long total = (long)M * (K / QK_NVFP4_SUB);
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    k_quant_acts<<<(unsigned)blocks, threads, 0, stream>>>(
        src1_sorted, row_scale, (uint8_t *)a_e2m1, (uint8_t *)sfa,
        row_expert, expert_offsets, sf_offsets, M, K, numKTiles);
}

extern "C" void ggml_cuda_bf16_to_f32_buf(
        const void * src_bf16, float * dst, int64_t n, cudaStream_t stream) {
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    k_bf16_to_f32<<<(unsigned)blocks, threads, 0, stream>>>(
        (const uint16_t *)src_bf16, dst, n);
}

extern "C" void ggml_cuda_bf16_to_f32_rowscaled(
        const void * src_bf16, float * dst, const float * row_scale,
        int M, int N, cudaStream_t stream) {
    const int threads = 256;
    const int64_t blocks = ((int64_t)M * N + threads - 1) / threads;
    k_bf16_to_f32_rowscaled<<<(unsigned)blocks, threads, 0, stream>>>(
        (const uint16_t *)src_bf16, dst, row_scale, M, N);
}

// ============================================================================
// zero-sync prefill path (2026-07-02)
// ============================================================================

// sf_offsets[e] = sum_{j<e} round_up(Me_j, 128), from expert_bounds, on device.
// E <= 1024 assumed (single block scan; E is 128 for Qwen3-MoE).
__global__ void k_sf_offsets(const int32_t * expert_bounds, int32_t * sf_offsets, int E) {
    __shared__ int32_t vals[1024];
    const int e = threadIdx.x;
    if (e < E) {
        const int me = expert_bounds[e + 1] - expert_bounds[e];
        vals[e] = ((me + 127) / 128) * 128;
    }
    __syncthreads();
    if (e == 0) {
        int32_t acc = 0;
        for (int j = 0; j < E; ++j) { sf_offsets[j] = acc; acc += vals[j]; }
    }
}

extern "C" void ggml_cuda_nvfp4_sf_offsets(
        const int32_t * expert_bounds, int32_t * sf_offsets, int E, cudaStream_t stream) {
    k_sf_offsets<<<1, (E + 31) / 32 * 32, 0, stream>>>(expert_bounds, sf_offsets, E);
}

// Fused gather + row-amax + quantize. One CUDA block per compact row m:
//   pass 1: block-stride load of src1 row via ids_src1[m], cache in smem, reduce amax
//   pass 2: quantize cached row (16-elem sub-blocks per thread) in the a/g domain
// Binary-search expert id from expert_bounds (E small, log2(128)=7 steps).
__global__ void k_gather_quant(
        const float * __restrict__ src1, const int32_t * __restrict__ ids_src1,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ sf_offsets,
        float * __restrict__ row_scale, uint8_t * __restrict__ a_e2m1, uint8_t * __restrict__ sfa,
        int E, int Mtot, int K, int64_t s11, int numKTiles) {
    const int m = blockIdx.x;
    if (m >= Mtot) return;

    extern __shared__ float srow[];               // K floats + 32 reduce slots
    float * red = srow + K;

    // ids_src1 == nullptr -> dense (identity gather): row m reads src1 row m
    const float * xrow = src1 + (int64_t) (ids_src1 ? ids_src1[m] : m) * s11;

    // pass 1: gather + amax (vectorized float4; K % 64 == 0 guaranteed by caller,
    // but xrow alignment depends on s11 — fall back to scalar if not 16B-aligned)
    float amax = 0.f;
    if (((uintptr_t) xrow & 15) == 0) {
        const float4 * x4 = (const float4 *) xrow;
        float4 * s4 = (float4 *) srow;
        for (int k = threadIdx.x; k < K / 4; k += blockDim.x) {
            const float4 v = x4[k];
            s4[k] = v;
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    } else {
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            const float v = xrow[k];
            srow[k] = v;
            amax = fmaxf(amax, fabsf(v));
        }
    }
    // block reduce (warp shuffle + smem across warps)
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        const int nwarp = (blockDim.x + 31) >> 5;
        float a = threadIdx.x < nwarp ? red[threadIdx.x] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));
        if (threadIdx.x == 0) red[0] = a;
    }
    __syncthreads();
    const float g = red[0] > 0.f ? red[0] / 2688.0f : 1.0f;
    if (threadIdx.x == 0) row_scale[m] = g;
    const float invg = 1.0f / g;

    // expert id for row m: binary search in expert_bounds
    int lo = 0, hi = E - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (expert_bounds[mid] <= m) lo = mid; else hi = mid - 1;
    }
    const int e = lo;
    const int r = m - expert_bounds[e];
    uint8_t * sfa_e   = sfa + (size_t) sf_offsets[e] * (K / QK_NVFP4_SUB);
    uint8_t * out_row = a_e2m1 + (size_t) m * (K / 2);

    // pass 2: quantize sub-blocks of 16
    const int nsubK = K / QK_NVFP4_SUB;
    for (int gsub = threadIdx.x; gsub < nsubK; gsub += blockDim.x) {
        const float * xb = srow + gsub * QK_NVFP4_SUB;
        float sub_amax = 0.f;
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB; j++)
            sub_amax = fmaxf(sub_amax, fabsf(xb[j] * invg));
        const uint8_t ue = d_fp32_to_ue4m3(sub_amax / 6.0f);
        const float scale = d_ue4m3_to_fp32_std(ue);
        sfa_e[d_sf_swizzle(r, gsub, numKTiles)] = ue;
        const int out_base = gsub * (QK_NVFP4_SUB / 2);
        const float invs = invg / scale;   // x * invg / scale in one mul
#if __CUDA_ARCH__ >= 1200
        // hardware e2m1 pair convert: byte = cvt(hi)<<4 | cvt(lo)
#pragma unroll
        for (int b = 0; b < QK_NVFP4_SUB / 2; b++) {
            uint32_t byte_;
            asm volatile("{\n\t.reg .b8 b8;\n\t"
                         "cvt.rn.satfinite.e2m1x2.f32 b8, %2, %1;\n\t"
                         "cvt.u32.u8 %0, b8;\n\t}"
                         : "=r"(byte_)
                         : "f"(xb[2 * b] * invs), "f"(xb[2 * b + 1] * invs));
            out_row[out_base + b] = (uint8_t) byte_;
        }
#else
#pragma unroll
        for (int b = 0; b < QK_NVFP4_SUB / 2; b++) {
            const uint8_t c0 = d_e2m1_code(xb[2 * b]     * invg, scale);
            const uint8_t c1 = d_e2m1_code(xb[2 * b + 1] * invg, scale);
            out_row[out_base + b] = c0 | (c1 << 4);
        }
#endif
    }
}

// Fused SwiGLU + NVFP4 requant on SORTED rows (whole-FFN fusion path).
// Row m: x[j] = silu(g*gate[m][j]) * (g*up[m][j]); then quantize x to NVFP4
// exactly like k_gather_quant pass 2 (same swizzle, same sf_offsets layout,
// K2 = N_gu). Replaces: scatter(gate) + scatter(up) + silu + gather_quant.
__global__ void k_swiglu_quant(
        const uint16_t * __restrict__ dg, const uint16_t * __restrict__ du,
        const float * __restrict__ row_scale1,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ sf_offsets,
        float * __restrict__ row_scale2, uint8_t * __restrict__ a_e2m1, uint8_t * __restrict__ sfa,
        int E, int Mtot, int K, int numKTiles) {
    const int m = blockIdx.x;
    if (m >= Mtot) return;
    extern __shared__ float srow[];               // K floats + 32 reduce slots
    float * red = srow + K;
    const float g1 = row_scale1[m];
    const uint16_t * grow_ = dg + (size_t) m * K;
    const uint16_t * urow_ = du + (size_t) m * K;
    float amax = 0.f;
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        uint32_t bg = (uint32_t) grow_[j] << 16, bu = (uint32_t) urow_[j] << 16;
        float xg, xu; memcpy(&xg, &bg, 4); memcpy(&xu, &bu, 4);
        xg *= g1; xu *= g1;
        const float v = (xg / (1.0f + expf(-xg))) * xu;   // silu(gate)*up
        srow[j] = v;
        amax = fmaxf(amax, fabsf(v));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        const int nwarp = (blockDim.x + 31) >> 5;
        float a = threadIdx.x < nwarp ? red[threadIdx.x] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));
        if (threadIdx.x == 0) red[0] = a;
    }
    __syncthreads();
    const float g = red[0] > 0.f ? red[0] / 2688.0f : 1.0f;
    if (threadIdx.x == 0) row_scale2[m] = g;
    const float invg = 1.0f / g;

    int lo = 0, hi = E - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (expert_bounds[mid] <= m) lo = mid; else hi = mid - 1;
    }
    const int e = lo;
    const int r = m - expert_bounds[e];
    uint8_t * sfa_e   = sfa + (size_t) sf_offsets[e] * (K / QK_NVFP4_SUB);
    uint8_t * out_row = a_e2m1 + (size_t) m * (K / 2);

    const int nsubK = K / QK_NVFP4_SUB;
    for (int gsub = threadIdx.x; gsub < nsubK; gsub += blockDim.x) {
        const float * xb = srow + gsub * QK_NVFP4_SUB;
        float sub_amax = 0.f;
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB; j++)
            sub_amax = fmaxf(sub_amax, fabsf(xb[j] * invg));
        const uint8_t ue = d_fp32_to_ue4m3(sub_amax / 6.0f);
        const float scale = d_ue4m3_to_fp32_std(ue);
        sfa_e[d_sf_swizzle(r, gsub, numKTiles)] = ue;
        const int out_base = gsub * (QK_NVFP4_SUB / 2);
#if __CUDA_ARCH__ >= 1200
        const float invs = invg / scale;
#pragma unroll
        for (int b = 0; b < QK_NVFP4_SUB / 2; b++) {
            uint32_t byte_;
            asm volatile("{\n\t.reg .b8 b8;\n\t"
                         "cvt.rn.satfinite.e2m1x2.f32 b8, %2, %1;\n\t"
                         "cvt.u32.u8 %0, b8;\n\t}"
                         : "=r"(byte_)
                         : "f"(xb[2 * b] * invs), "f"(xb[2 * b + 1] * invs));
            out_row[out_base + b] = (uint8_t) byte_;
        }
#else
#pragma unroll
        for (int b = 0; b < QK_NVFP4_SUB / 2; b++) {
            const uint8_t c0 = d_e2m1_code(xb[2 * b]     * invg, scale);
            const uint8_t c1 = d_e2m1_code(xb[2 * b + 1] * invg, scale);
            out_row[out_base + b] = c0 | (c1 << 4);
        }
#endif
    }
}

extern "C" void ggml_cuda_nvfp4_swiglu_quant(
        const void * d_gate_bf16, const void * d_up_bf16, const float * row_scale1,
        const int32_t * expert_bounds, const int32_t * sf_offsets,
        float * row_scale2, void * a_e2m1, void * sfa,
        int E, int Mtot, int K, cudaStream_t stream) {
    const int threads = 256;
    const int numKTiles = (K + 63) / 64;
    const size_t smem = (size_t) K * sizeof(float) + 32 * sizeof(float);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(k_swiglu_quant, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    }
    k_swiglu_quant<<<(unsigned) Mtot, threads, smem, stream>>>(
        (const uint16_t *) d_gate_bf16, (const uint16_t *) d_up_bf16, row_scale1,
        expert_bounds, sf_offsets, row_scale2,
        (uint8_t *) a_e2m1, (uint8_t *) sfa, E, Mtot, K, numKTiles);
}

extern "C" void ggml_cuda_nvfp4_gather_quant(
        const float * src1, const int32_t * ids_src1, const int32_t * expert_bounds,
        const int32_t * sf_offsets, float * row_scale, void * a_e2m1, void * sfa,
        int E, int Mtot, int K, int64_t s11, cudaStream_t stream) {
    const int threads = 256;
    const int numKTiles = (K + 63) / 64;
    const size_t smem = (size_t) K * sizeof(float) + 32 * sizeof(float);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(k_gather_quant, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    }
    k_gather_quant<<<(unsigned) Mtot, threads, smem, stream>>>(
        src1, ids_src1, expert_bounds, sf_offsets, row_scale,
        (uint8_t *) a_e2m1, (uint8_t *) sfa, E, Mtot, K, s11, numKTiles);
}

// scatter + rescale + bf16->f32: one block per compact row, vectorized 4-wide.
// N % 4 == 0 always holds here (N is a multiple of 128 per _supported()).
// accum=false: dst row = ids_dst[m], overwrite (plain scatter).
// accum=true : dst row = ids_dst[m]/accum_neu (token index), atomicAdd — fuses
//              the sum-over-experts (view+add chain) into the scatter. Caller
//              must zero dst first.
template <bool accum>
__global__ void k_scatter_rowscaled(
        const uint16_t * __restrict__ src, const float * __restrict__ row_scale,
        const int32_t * __restrict__ ids_dst, float * __restrict__ dst,
        int Mtot, int N, int64_t s1, const float * __restrict__ fuse_w, int accum_neu) {
    const int m = blockIdx.x;
    if (m >= Mtot) return;
    const int id = ids_dst[m];
    float g = row_scale[m];
    if (fuse_w) g *= fuse_w[id];   // fused routing-weight multiply
    const ushort4 * s4 = (const ushort4 *) (src + (size_t) m * N);
    float * drow = dst + (int64_t) (accum ? id / accum_neu : id) * s1;
    const bool aligned = ((uintptr_t) drow & 15) == 0;
    for (int n4 = threadIdx.x; n4 < N / 4; n4 += blockDim.x) {
        const ushort4 v = s4[n4];
        float4 f;
        uint32_t b;
        b = (uint32_t) v.x << 16; memcpy(&f.x, &b, 4); f.x *= g;
        b = (uint32_t) v.y << 16; memcpy(&f.y, &b, 4); f.y *= g;
        b = (uint32_t) v.z << 16; memcpy(&f.z, &b, 4); f.z *= g;
        b = (uint32_t) v.w << 16; memcpy(&f.w, &b, 4); f.w *= g;
        if (accum) {
            atomicAdd(drow + n4 * 4 + 0, f.x); atomicAdd(drow + n4 * 4 + 1, f.y);
            atomicAdd(drow + n4 * 4 + 2, f.z); atomicAdd(drow + n4 * 4 + 3, f.w);
        } else if (aligned) {
            ((float4 *) drow)[n4] = f;
        } else {
            drow[n4 * 4 + 0] = f.x; drow[n4 * 4 + 1] = f.y;
            drow[n4 * 4 + 2] = f.z; drow[n4 * 4 + 3] = f.w;
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel routing build (2026-07-02): replaces mm_ids_helper for the CUTLASS
// MoE paths. mm_ids_helper uses ONE WARP PER EXPERT serially scanning all
// tokens (128 warps total on a 188-SM GPU, ~52us). This is histogram -> scan
// -> scatter over Mtot threads (~5us). Within-expert row order becomes
// arbitrary (atomic cursors) — harmless: each GEMM row is independent and both
// scatter and gather-accum address rows through the same maps.
// ---------------------------------------------------------------------------
__global__ void k_route_hist(
        const int32_t * __restrict__ ids, int32_t * __restrict__ counts,
        int n_tokens, int neu, int si1, int E) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tokens * neu) return;
    const int it  = i / neu;
    const int iex = i % neu;
    const int e   = ids[it * si1 + iex];
    if (e < 0 || e >= E) return; // skip padded/garbage expert ids (OOB write guard)
    atomicAdd(&counts[e], 1);
}
// single block, E <= 1024: exclusive scan counts -> bounds[0..E], zero cursors
__global__ void k_route_scan(
        const int32_t * __restrict__ counts, int32_t * __restrict__ bounds,
        int32_t * __restrict__ cursor, int E) {
    __shared__ int32_t s[1025];
    const int e = threadIdx.x;
    if (e <= E) s[e] = 0;
    __syncthreads();
    // simple Hillis-Steele over E entries (E=128 typical: 7 steps)
    int v = e < E ? counts[e] : 0;
    s[e + 1] = v;
    __syncthreads();
    for (int off = 1; off <= E; off <<= 1) {
        int add = (e + 1 > off) ? s[e + 1 - off] : 0;
        __syncthreads();
        s[e + 1] += add;
        __syncthreads();
    }
    if (e <= E) bounds[e] = s[e];
    if (e < E)  cursor[e] = 0;
}
__global__ void k_route_scatter(
        const int32_t * __restrict__ ids, const int32_t * __restrict__ bounds,
        int32_t * __restrict__ cursor, int32_t * __restrict__ ids_src1,
        int32_t * __restrict__ ids_dst, int n_tokens, int neu, int nchannels_y,
        int si1, int sis1, int E) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tokens * neu) return;
    const int it  = i / neu;
    const int iex = i % neu;
    const int e   = ids[it * si1 + iex];
    if (e < 0 || e >= E) return; // must mirror k_route_hist guard
    const int pos = bounds[e] + atomicAdd(&cursor[e], 1);
    ids_src1[pos] = it * sis1 + iex % nchannels_y;
    ids_dst [pos] = it * neu  + iex;
}
extern "C" void ggml_cuda_moe_build_routing(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst,
        int32_t * bounds, int32_t * scratch /* 2*E ints: counts+cursor */,
        int E, int n_tokens, int neu, int nchannels_y, int si1, int sis1,
        cudaStream_t stream) {
    int32_t * counts = scratch;
    int32_t * cursor = scratch + E;
    cudaMemsetAsync(counts, 0, E * sizeof(int32_t), stream);
    const int Mtot = n_tokens * neu;
    const int nb = (Mtot + 255) / 256;
    k_route_hist<<<nb, 256, 0, stream>>>(ids, counts, n_tokens, neu, si1, E);
    int sb = 32; while (sb < E + 1) sb <<= 1;   // block >= E+1 threads, pow2
    k_route_scan<<<1, sb, 0, stream>>>(counts, bounds, cursor, E);
    k_route_scatter<<<nb, 256, 0, stream>>>(ids, bounds, cursor, ids_src1, ids_dst,
                                            n_tokens, neu, nchannels_y, si1, sis1, E);
}


// Invert the compact->flat permutation: inv[ids_dst[m]] = m.
// ids_dst is a bijection over [0, Mtot) (every token owns exactly neu slots).
__global__ void k_moe_invert_ids(const int32_t * __restrict__ ids_dst, int32_t * __restrict__ inv, int Mtot) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < Mtot) {
        inv[ids_dst[m]] = m;
    }
}

extern "C" void ggml_cuda_moe_invert_ids(const int32_t * ids_dst, int32_t * inv, int Mtot, cudaStream_t stream) {
    k_moe_invert_ids<<<(unsigned) ((Mtot + 255)/256), 256, 0, stream>>>(ids_dst, inv, Mtot);
}

// GATHER replacement for the accum scatter: one block per TOKEN. Reads the
// token's neu compact rows via the inverse permutation, sums in registers,
// writes each dst element exactly once (coalesced float4, no atomics, no
// dst pre-zeroing). ~5x faster than the atomicAdd scatter at prefill sizes.
// bf16 src variant (NVFP4 path): gain = row_scale[m] * fuse_w[flat].
__global__ void k_gather_accum_rowscaled(
        const uint16_t * __restrict__ src, const float * __restrict__ row_scale,
        const int32_t * __restrict__ inv, float * __restrict__ dst,
        int n_tokens, int N, int64_t s1, const float * __restrict__ fuse_w, int neu) {
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    __shared__ int   s_m[16];
    __shared__ float s_g[16];
    if (threadIdx.x < (unsigned) neu) {
        const int flat = t * neu + (int) threadIdx.x;
        const int m    = inv[flat];
        s_m[threadIdx.x] = m;
        float g = row_scale[m];
        if (fuse_w) g *= fuse_w[flat];
        s_g[threadIdx.x] = g;
    }
    __syncthreads();
    float * drow = dst + (int64_t) t * s1;
    const bool aligned = ((uintptr_t) drow & 15) == 0;
    for (int n4 = threadIdx.x; n4 < N / 4; n4 += blockDim.x) {
        float4 acc = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll 4
        for (int e = 0; e < neu; ++e) {
            const ushort4 v = ((const ushort4 *) (src + (size_t) s_m[e] * N))[n4];
            const float g = s_g[e];
            uint32_t b; float f;
            b = (uint32_t) v.x << 16; memcpy(&f, &b, 4); acc.x += f * g;
            b = (uint32_t) v.y << 16; memcpy(&f, &b, 4); acc.y += f * g;
            b = (uint32_t) v.z << 16; memcpy(&f, &b, 4); acc.z += f * g;
            b = (uint32_t) v.w << 16; memcpy(&f, &b, 4); acc.w += f * g;
        }
        if (aligned) {
            ((float4 *) drow)[n4] = acc;
        } else {
            drow[n4*4+0] = acc.x; drow[n4*4+1] = acc.y;
            drow[n4*4+2] = acc.z; drow[n4*4+3] = acc.w;
        }
    }
}

extern "C" void ggml_cuda_nvfp4_scatter_rowscaled(
        const void * d_bf16, const float * row_scale, const int32_t * ids_dst,
        float * dst, int Mtot, int N, int64_t s1, const float * fuse_w, int accum_neu,
        cudaStream_t stream) {
    const int threads = 256;
    if (accum_neu > 0) {
        k_scatter_rowscaled<true><<<(unsigned) Mtot, threads, 0, stream>>>(
            (const uint16_t *) d_bf16, row_scale, ids_dst, dst, Mtot, N, s1, fuse_w, accum_neu);
    } else {
        k_scatter_rowscaled<false><<<(unsigned) Mtot, threads, 0, stream>>>(
            (const uint16_t *) d_bf16, row_scale, ids_dst, dst, Mtot, N, s1, fuse_w, accum_neu);
    }
}

extern "C" void ggml_cuda_nvfp4_gather_accum_rowscaled(
        const void * d_bf16, const float * row_scale, const int32_t * inv,
        float * dst, int n_tokens, int N, int64_t s1, const float * fuse_w, int neu,
        cudaStream_t stream) {
    k_gather_accum_rowscaled<<<(unsigned) n_tokens, 256, 0, stream>>>(
        (const uint16_t *) d_bf16, row_scale, inv, dst, n_tokens, N, s1, fuse_w, neu);
}
// ============================================================================
// F16 MoE prefill glue (2026-07-02): gather f32->f16, scatter f32->f32
// ============================================================================

// dst_f16[m,k] = (half) src1[ids_src1[m]*s11 + k]; one block per compact row
__global__ void k_gather_f32_to_f16(
        const float * __restrict__ src1, const int32_t * __restrict__ ids_src1,
        __half * __restrict__ dst, int Mtot, int K, int64_t s11) {
    const int m = blockIdx.x;
    if (m >= Mtot) return;
    const float * xrow = src1 + (int64_t) ids_src1[m] * s11;
    __half * drow = dst + (size_t) m * K;
    if ((((uintptr_t) xrow | (uintptr_t) drow) & 15) == 0 && (K & 7) == 0) {
        const float4 * x4 = (const float4 *) xrow;
        // 8 halves = 16 bytes per store: process two float4 loads per iter
        for (int i = threadIdx.x; i < K / 8; i += blockDim.x) {
            const float4 a = x4[2 * i];
            const float4 b = x4[2 * i + 1];
            __half2 h[4] = {
                __floats2half2_rn(a.x, a.y), __floats2half2_rn(a.z, a.w),
                __floats2half2_rn(b.x, b.y), __floats2half2_rn(b.z, b.w) };
            ((uint4 *) drow)[i] = *(const uint4 *) h;
        }
    } else {
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            drow[k] = __float2half_rn(xrow[k]);
        }
    }
}

extern "C" void ggml_cuda_moe_gather_f32_to_f16(
        const float * src1, const int32_t * ids_src1, void * dst_f16,
        int Mtot, int K, int64_t s11, cudaStream_t stream) {
    k_gather_f32_to_f16<<<(unsigned) Mtot, 256, 0, stream>>>(
        src1, ids_src1, (__half *) dst_f16, Mtot, K, s11);
}

// dst[ids_dst[m]*s1 + n] = src[m*N + n]; one block per compact row, float4
template <bool accum>
__global__ void k_scatter_f32(
        const float * __restrict__ src, const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst, int Mtot, int N, int64_t s1,
        const float * __restrict__ fuse_w, int accum_neu) {
    const int m = blockIdx.x;
    if (m >= Mtot) return;
    const int id = ids_dst[m];
    const float g = fuse_w ? fuse_w[id] : 1.0f;  // fused routing-weight multiply
    const float * srow = src + (size_t) m * N;
    float * drow = dst + (int64_t) (accum ? id / accum_neu : id) * s1;
    if (accum) {
        for (int n = threadIdx.x; n < N; n += blockDim.x) {
            atomicAdd(drow + n, srow[n] * g);
        }
    } else if ((((uintptr_t) srow | (uintptr_t) drow) & 15) == 0 && (N & 3) == 0) {
        for (int i = threadIdx.x; i < N / 4; i += blockDim.x) {
            float4 v = ((const float4 *) srow)[i];
            v.x *= g; v.y *= g; v.z *= g; v.w *= g;
            ((float4 *) drow)[i] = v;
        }
    } else {
        for (int n = threadIdx.x; n < N; n += blockDim.x) {
            drow[n] = srow[n] * g;
        }
    }
}

extern "C" void ggml_cuda_moe_scatter_f32(
        const float * src, const int32_t * ids_dst, float * dst,
        int Mtot, int N, int64_t s1, const float * fuse_w, int accum_neu, cudaStream_t stream) {
    if (accum_neu > 0) {
        k_scatter_f32<true><<<(unsigned) Mtot, 256, 0, stream>>>(src, ids_dst, dst, Mtot, N, s1, fuse_w, accum_neu);
    } else {
        k_scatter_f32<false><<<(unsigned) Mtot, 256, 0, stream>>>(src, ids_dst, dst, Mtot, N, s1, fuse_w, accum_neu);
    }
}

// f32-src gather-accum variant (F16 MoE path). Same inverse-permutation
// no-atomic design as k_gather_accum_rowscaled.
__global__ void k_gather_accum_f32(
        const float * __restrict__ src, const int32_t * __restrict__ inv,
        float * __restrict__ dst, int n_tokens, int N, int64_t s1,
        const float * __restrict__ fuse_w, int neu) {
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    __shared__ int   s_m[16];
    __shared__ float s_g[16];
    if (threadIdx.x < (unsigned) neu) {
        const int flat = t * neu + (int) threadIdx.x;
        s_m[threadIdx.x] = inv[flat];
        s_g[threadIdx.x] = fuse_w ? fuse_w[flat] : 1.0f;
    }
    __syncthreads();
    float * drow = dst + (int64_t) t * s1;
    if ((((uintptr_t) src | (uintptr_t) drow) & 15) == 0 && (N & 3) == 0) {
        for (int n4 = threadIdx.x; n4 < N / 4; n4 += blockDim.x) {
            float4 acc = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll 4
            for (int e = 0; e < neu; ++e) {
                const float4 v = ((const float4 *) (src + (size_t) s_m[e] * N))[n4];
                const float g = s_g[e];
                acc.x += v.x * g; acc.y += v.y * g; acc.z += v.z * g; acc.w += v.w * g;
            }
            ((float4 *) drow)[n4] = acc;
        }
    } else {
        for (int n = threadIdx.x; n < N; n += blockDim.x) {
            float acc = 0.f;
            for (int e = 0; e < neu; ++e) {
                acc += src[(size_t) s_m[e] * N + n] * s_g[e];
            }
            drow[n] = acc;
        }
    }
}

extern "C" void ggml_cuda_moe_gather_accum_f32(
        const float * src, const int32_t * inv, float * dst,
        int n_tokens, int N, int64_t s1, const float * fuse_w, int neu, cudaStream_t stream) {
    k_gather_accum_f32<<<(unsigned) n_tokens, 256, 0, stream>>>(src, inv, dst, n_tokens, N, s1, fuse_w, neu);
}
