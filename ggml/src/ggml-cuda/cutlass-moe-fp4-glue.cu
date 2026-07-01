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
