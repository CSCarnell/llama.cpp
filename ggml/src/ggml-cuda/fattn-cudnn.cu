// FCFA-cuDNN: route large-batch prefill FLASH_ATTN_EXT through cuDNN's SDPA
// engine (Blackwell FA3-class warp-specialized/TMA kernels). ~1.5-2.0x faster
// than the built-in mma-f16 kernel at production prefill shapes (see
// CUDA-DEDICATED-ENGINE-PLAN.md 2026-07-02 (f)).
//
// Constraints honored here (all proven by standalone protos):
//  * cuDNN is only fast with its NATIVE causal-band + padding mask; feeding
//    ggml's additive mask as a bias tensor is 2.7x slower than the band. So we
//    PROVE the mask is exactly a bottom-right causal band with a cheap GPU
//    probe (one ~20us pass per ubatch, result cached across all layers of the
//    same graph eval) and fall back to the built-in kernels otherwise.
//  * K/V are consumed IN PLACE from the ggml KV cache via generic strides.
//  * Q is staged f32 -> f16 (cuDNN wants matching io types); O is staged f16
//    and converted to the f32 ggml dst (cuDNN's f32-output path was
//    numerically wrong in prototyping - do not revisit).
//  * Graphs are cached per shape; out-of-view KV rows covered by a dim bucket
//    are masked by the padding mask (NaN-poison proven inert in proto5).

#include "common.cuh"
#include "fattn-cudnn.cuh"

#ifdef GGML_CUDA_USE_CUDNN

#include <cudnn_frontend.h>
#include <cuda_fp8.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace fe = cudnn_frontend;

// ---------------------------------------------------------------- kernels

// Q staging: strided f32 [d, n_q, h_q] view -> contiguous f16 interleaved [s][h][d].
static __global__ void k_fcfa_q_f32_to_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t d, const int64_t n_q, const int64_t h_q,
        const int64_t nb1, const int64_t nb2) { // byte strides: nb1 = q pos, nb2 = head
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= d*n_q*h_q) {
        return;
    }
    const int64_t x = i % d;
    const int64_t h = (i / d) % h_q;
    const int64_t s = i / (d*h_q);
    const float * p = (const float *) (src + s*nb1 + h*nb2);
    dst[i] = __float2half(p[x]);
}

// O staging: flat f16 -> f32 (layouts already identical).
static __global__ void k_fcfa_o_f16_to_f32(const half * __restrict__ src, float * __restrict__ dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}


// ================= FP8 (E4M3) attention staging (GGML_CUDA_CUDNN_FA_FP8) =====
// Per-tensor dynamic amax scaling: amax(|X|) -> scale = 448/amax quantizes to
// E4M3, descale = amax/448 restores magnitude inside cuDNN. Softmax probs P in
// [0,1] use a fixed scale_s=448. O is emitted in HALF (scale_o=1) so there is
// no output-side FP8 precision loss and the existing o_f16->f32 stage is reused.
#include <cooperative_groups.h>

#define FCFA_E4M3_MAX 448.0f

// atomicMax on the int bit-pattern is monotonic for non-negative floats.
static __device__ __forceinline__ void fcfa_atomic_max_pos(float * addr, float v) {
    atomicMax((int *) addr, __float_as_int(v));
}

// Fused single-pass staging: amax over a strided [d,n,h] view, then quantize the
// SAME view to contiguous E4M3 [s][h][d] — one cooperative-groups launch per
// tensor (grid.sync() between the amax and quantize passes). Writes the cuDNN
// descale scalar inline. Replaces memset+3*amax+make_scales+3*quantize
// (8 launches) with 3 launches. Exact per-tensor amax (no stale/delayed scale).
//   src_is_f32=1: src is char* base with BYTE strides nb1/nb2 (Q, from F32)
//   src_is_f32=0: src is half* base (already band-offset), ELEMENT strides (K/V)
//   extras: 0=K (descale only); 1=Q (also ds[3]=1/448, ds[4]=448);
//           2=V (also ds[5]=448/amax_v -> scale_o, O reuses V range)
static __global__ void k_fcfa_stage_fused(
        const char * __restrict__ src, __nv_fp8_e4m3 * __restrict__ dst,
        const int64_t d, const int64_t n, const int64_t h,
        const int64_t nb1, const int64_t nb2, const int src_is_f32,
        float * __restrict__ amax_scratch, float * __restrict__ descale_out,
        const int extras, float * __restrict__ dsb) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    const int64_t total  = d*n*h;
    const int64_t stride = (int64_t) gridDim.x*blockDim.x;
    const int64_t gtid   = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    if (gtid == 0) *amax_scratch = 0.0f;
    grid.sync();

    // pass 1: amax over the strided view
    float m = 0.0f;
    for (int64_t i = gtid; i < total; i += stride) {
        const int64_t x = i % d, hh = (i / d) % h, s = i / (d*h);
        const float val = src_is_f32
            ? ((const float *) (src + s*nb1 + hh*nb2))[x]
            : __half2float(((const half *) src)[s*nb1 + hh*nb2 + x]);
        m = fmaxf(m, fabsf(val));
    }
    __shared__ float sm[256];
    sm[threadIdx.x] = m; __syncthreads();
    for (int o = blockDim.x/2; o > 0; o >>= 1) {
        if (threadIdx.x < (unsigned) o) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x+o]);
        __syncthreads();
    }
    if (threadIdx.x == 0) fcfa_atomic_max_pos(amax_scratch, sm[0]);
    grid.sync();

    // pass 2: quantize the same view with the global amax
    const float amax   = fmaxf(*amax_scratch, 1e-6f);
    const float qscale = FCFA_E4M3_MAX / amax;
    for (int64_t i = gtid; i < total; i += stride) {
        const int64_t x = i % d, hh = (i / d) % h, s = i / (d*h);
        const float val = src_is_f32
            ? ((const float *) (src + s*nb1 + hh*nb2))[x]
            : __half2float(((const half *) src)[s*nb1 + hh*nb2 + x]);
        dst[i] = __nv_fp8_e4m3(val * qscale); // dst is contiguous [s][h][x] == i
    }
    if (gtid == 0) {
        *descale_out = amax / FCFA_E4M3_MAX;
        if (extras == 1) { dsb[3] = 1.0f / FCFA_E4M3_MAX; dsb[4] = FCFA_E4M3_MAX; }
        if (extras == 2) { dsb[5] = FCFA_E4M3_MAX / amax; }
    }
}

// Cooperative launch helper: sizes the grid to the max co-resident blocks so
// grid.sync() is legal, then grid-strides over the whole view.
static void fcfa_launch_stage(
        const void * src, __nv_fp8_e4m3 * dst,
        int64_t d, int64_t n, int64_t h, int64_t nb1, int64_t nb2, int src_is_f32,
        float * amax_scratch, float * descale_out, int extras, float * dsb,
        cudaStream_t stream) {
    static int grid_blocks = 0;
    if (grid_blocks == 0) {
        int dev = 0; cudaGetDevice(&dev);
        int nsm = 0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        int per_sm = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, (const void *) k_fcfa_stage_fused, 256, 0);
        grid_blocks = nsm * (per_sm > 0 ? per_sm : 1);
    }
    const char * csrc = (const char *) src;
    void * args[] = { (void *) &csrc, (void *) &dst, (void *) &d, (void *) &n, (void *) &h,
                      (void *) &nb1, (void *) &nb2, (void *) &src_is_f32,
                      (void *) &amax_scratch, (void *) &descale_out, (void *) &extras, (void *) &dsb };
    cudaLaunchCooperativeKernel((const void *) k_fcfa_stage_fused,
        dim3((unsigned) grid_blocks), dim3(256), args, 0, stream);
}

// Dequantize E4M3 O [s][h][dv] -> f32 dst. descale = 1/scale_o = amax_v/448 = ds[2].
static __global__ void k_fcfa_o_fp8_to_f32(const __nv_fp8_e4m3 * __restrict__ src,
        float * __restrict__ dst, const int64_t n, const float * __restrict__ descale) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (float) src[i] * descale[0];
    }
}


// allowed count. "Allowed" = mask value 0; any finite non-zero value (ALiBi
// etc.) poisons the count so the band check fails. One block per row.
// A row is a *contiguous* run iff cnt == hi - lo + 1.
static __global__ void k_fcfa_mask_probe(
        const half * __restrict__ mask, const int64_t n_kv, const int64_t row_stride,
        int32_t * __restrict__ lo_out, int32_t * __restrict__ hi_out, int32_t * __restrict__ cnt_out) {
    const half * row = mask + (int64_t) blockIdx.x*row_stride;
    int32_t lo  = INT32_MAX;
    int32_t hi  = -1;
    int32_t cnt = 0;
    for (int64_t j = threadIdx.x; j < n_kv; j += blockDim.x) {
        const float v = __half2float(row[j]);
        if (!(v < -1e30f)) { // allowed (not -inf)
            cnt++;
            if ((int32_t) j > hi) {
                hi = (int32_t) j;
            }
            if ((int32_t) j < lo) {
                lo = (int32_t) j;
            }
            if (v != 0.0f) {
                cnt = INT32_MIN/2; // non-band mask -> poison
            }
        }
    }
    __shared__ int32_t s_lo [256];
    __shared__ int32_t s_hi [256];
    __shared__ int32_t s_cnt[256];
    s_lo [threadIdx.x] = lo;
    s_hi [threadIdx.x] = hi;
    s_cnt[threadIdx.x] = cnt;
    __syncthreads();
    for (int off = blockDim.x/2; off > 0; off >>= 1) {
        if (threadIdx.x < (unsigned) off) {
            s_lo [threadIdx.x] = min(s_lo[threadIdx.x], s_lo[threadIdx.x + off]);
            s_hi [threadIdx.x] = max(s_hi[threadIdx.x], s_hi[threadIdx.x + off]);
            // saturating add so the poison survives reduction:
            const int64_t c = (int64_t) s_cnt[threadIdx.x] + s_cnt[threadIdx.x + off];
            s_cnt[threadIdx.x] = c < INT32_MIN/2 ? INT32_MIN/2 : (int32_t) c;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        lo_out [blockIdx.x] = s_lo[0];
        hi_out [blockIdx.x] = s_hi[0];
        cnt_out[blockIdx.x] = s_cnt[0];
    }
}

// Write the two int32 sequence lengths device-side (avoids tiny H2D copies).
static __global__ void k_fcfa_set_seq(int32_t * dst, const int32_t seq_q, const int32_t seq_kv) {
    dst[0] = seq_q;
    dst[1] = seq_kv;
}

// ---------------------------------------------------------------- state

struct fcfa_graph_key {
    int64_t n_q, n_kv, h_q, h_k, d, dv;
    int64_t q_nb1, q_nb2;       // not really needed (Q staged contig) but keep dst variants apart
    int64_t k_nb1, k_nb2, v_nb1, v_nb2; // element strides of K/V views
    uint32_t scale_bits;
    bool operator<(const fcfa_graph_key & o) const {
        return memcmp(this, &o, sizeof(o)) < 0;
    }
};

struct fcfa_graph_entry {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t ws_size = 0;
    bool build_failed = false;
};

struct fcfa_state {
    cudnnHandle_t handle = nullptr;
    std::map<fcfa_graph_key, fcfa_graph_entry> * graphs = nullptr; // leaked on purpose (exit-order safety)
    // per-graph-eval mask verdict cache:
    uint64_t mask_gen = 0;
    const void * mask_ptr = nullptr;
    int64_t mask_n_q = -1, mask_n_kv = -1;
    bool mask_checked = false;
    bool mask_is_band = false;
    int32_t band_lo     = 0; // L: constant first allowed key (kv-unified slot offset)
    int32_t band_n_past = 0; // D: row i allows [L, L+D+i]
    std::vector<int32_t> h_probe;
};

static fcfa_state & fcfa_get_state() {
    static fcfa_state st;
    if (st.graphs == nullptr) {
        st.graphs = new std::map<fcfa_graph_key, fcfa_graph_entry>();
    }
    return st;
}

static std::atomic<uint64_t> g_fcfa_gen{1};

void ggml_cuda_fattn_cudnn_begin_graph(void) {
    g_fcfa_gen.fetch_add(1, std::memory_order_relaxed);
}

static bool fcfa_log_enabled() {
    static const bool en = getenv("GGML_CUDA_CUDNN_FA_LOG") != nullptr;
    return en;
}

// ---------------------------------------------------------------- impl

enum fcfa_uids : int64_t { UID_Q = 1, UID_K, UID_V, UID_O, UID_SEQ_Q = 7, UID_SEQ_KV,
    // FP8 path scalars/outputs
    UID_DQ = 10, UID_DK, UID_DV, UID_DS, UID_SS, UID_SO, UID_AMAX_S, UID_AMAX_O };

// ---------------------------------------------------------------- FP8 path
// Isolated E4M3 attention path (GGML_CUDA_CUDNN_FA_FP8). Own graph cache so the
// validated HALF path is byte-for-byte untouched. Q/K/V are dynamically
// per-tensor amax-scaled into E4M3, cuDNN runs sdpa_fp8, O is emitted HALF
// (scale_o=1) and dequantized with the shared o_f16->f32 stage.
static bool fcfa_run_fp8(
        ggml_backend_cuda_context & ctx, fcfa_state & st, ggml_tensor * dst,
        const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V,
        int32_t band_lo, int32_t seq_kv, int64_t dim_kv, float scale,
        int64_t n_q, int64_t h_q, int64_t h_k, int64_t d, int64_t dv, cudaStream_t stream) {
    static std::map<fcfa_graph_key, fcfa_graph_entry> g_fp8_graphs;

    // FP8 sdpa engines support causal-bottom-right, NOT the padding-mask+seq_len
    // variant. So use the EXACT key window [band_lo, band_lo+seq_kv) with no
    // bucketing; bottom-right causal reproduces row i -> keys [0, n_past+i].
    const int64_t dim_kv_fp8 = seq_kv;
    GGML_UNUSED(dim_kv);
    const int64_t k_nb1 = (int64_t) (K->nb[1]/sizeof(half));
    const int64_t k_nb2 = (int64_t) (K->nb[2]/sizeof(half));
    const int64_t v_nb1 = (int64_t) (V->nb[1]/sizeof(half));
    const int64_t v_nb2 = (int64_t) (V->nb[2]/sizeof(half));

    fcfa_graph_key key = {};
    key.n_q  = n_q;  key.n_kv = dim_kv_fp8;
    key.h_q  = h_q;  key.h_k  = h_k;
    key.d    = d;    key.dv   = dv;
    key.q_nb1 = 0;   key.q_nb2 = 0;
    key.k_nb1 = k_nb1; key.k_nb2 = k_nb2;
    key.v_nb1 = v_nb1; key.v_nb2 = v_nb2;
    memcpy(&key.scale_bits, &scale, sizeof(scale));

    auto it = g_fp8_graphs.find(key);
    if (it == g_fp8_graphs.end()) {
        fcfa_graph_entry ent;
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::FP8_E4M3)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);
        // Q/K/V staged contiguous interleaved [s][h][d], E4M3.
        auto Qt = graph->tensor(fe::graph::Tensor_attributes().set_uid(UID_Q)
            .set_dim({1, h_q, n_q, d}).set_stride({d*n_q*h_q, d, d*h_q, 1}));
        auto Kt = graph->tensor(fe::graph::Tensor_attributes().set_uid(UID_K)
            .set_dim({1, h_k, dim_kv_fp8, d}).set_stride({d*dim_kv_fp8*h_k, d, d*h_k, 1}));
        auto Vt = graph->tensor(fe::graph::Tensor_attributes().set_uid(UID_V)
            .set_dim({1, h_k, dim_kv_fp8, d}).set_stride({d*dim_kv_fp8*h_k, d, d*h_k, 1}));
        auto mkscalar = [&](int64_t uid) {
            return graph->tensor(fe::graph::Tensor_attributes().set_uid(uid)
                .set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::FLOAT));
        };
        auto dq = mkscalar(UID_DQ), dk = mkscalar(UID_DK), dvv = mkscalar(UID_DV);
        auto ds = mkscalar(UID_DS), ss = mkscalar(UID_SS), so = mkscalar(UID_SO);

        auto opts = fe::graph::SDPA_fp8_attributes()
            .set_generate_stats(false)
            .set_causal_mask_bottom_right(true)
            .set_attn_scale(scale);

        auto [Ot, stats, amax_s, amax_o] = graph->sdpa_fp8(Qt, Kt, Vt, dq, dk, dvv, ds, ss, so, opts);
        GGML_UNUSED(stats);
        // O is E4M3 (io default), scaled by scale_o=448/amax_v; dequantized to f32.
        Ot->set_output(true).set_uid(UID_O)
          .set_dim({1, h_q, n_q, dv}).set_stride({dv*n_q*h_q, dv, dv*h_q, 1});
        amax_s->set_output(true).set_uid(UID_AMAX_S).set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1,1,1,1}).set_stride({1,1,1,1});
        amax_o->set_output(true).set_uid(UID_AMAX_O).set_data_type(fe::DataType_t::FLOAT)
          .set_dim({1,1,1,1}).set_stride({1,1,1,1});

        auto bst = graph->build(st.handle, {fe::HeurMode_t::A});
        if (!bst.is_good()) {
            ent.build_failed = true;
            if (fcfa_log_enabled()) {
                fprintf(stderr, "[FCFA-cuDNN-FP8] graph build FAILED (n_q=%ld dim_kv=%ld): %s\n",
                    (long) n_q, (long) dim_kv, bst.get_message().c_str());
            }
        } else {
            ent.graph = graph;
            (void) graph->get_workspace_size(ent.ws_size);
            if (fcfa_log_enabled()) {
                fprintf(stderr, "[FCFA-cuDNN-FP8] graph built n_q=%ld dim_kv=%ld ws=%.1f MB\n",
                    (long) n_q, (long) dim_kv, ent.ws_size/1048576.0);
            }
        }
        it = g_fp8_graphs.emplace(key, std::move(ent)).first;
    }
    if (it->second.build_failed) {
        return false;
    }
    const fcfa_graph_entry & ent = it->second;

    // ---- staging + execute ----
    ggml_cuda_pool_alloc<__nv_fp8_e4m3> q_fp8(ctx.pool(), d*n_q*h_q);
    ggml_cuda_pool_alloc<__nv_fp8_e4m3> k_fp8(ctx.pool(), d*dim_kv_fp8*h_k);
    ggml_cuda_pool_alloc<__nv_fp8_e4m3> v_fp8(ctx.pool(), d*dim_kv_fp8*h_k);
    ggml_cuda_pool_alloc<__nv_fp8_e4m3> o_fp8(ctx.pool(), dv*n_q*h_q);
    ggml_cuda_pool_alloc<float>         amax (ctx.pool(), 3);
    ggml_cuda_pool_alloc<float>         dsb  (ctx.pool(), 6); // descale q,k,v,s + scale_s + scale_o
    ggml_cuda_pool_alloc<float>         amo  (ctx.pool(), 2); // amax_s, amax_o sinks
    ggml_cuda_pool_alloc<char>          ws;
    if (ent.ws_size > 0) {
        ws.alloc(ctx.pool(), ent.ws_size);
    }

    const half * base_k = (const half *) ((const char *) K->data + (int64_t) band_lo*K->nb[1]);
    const half * base_v = (const half *) ((const char *) V->data + (int64_t) band_lo*V->nb[1]);

    // Fused single-pass staging: amax+quantize per tensor in ONE cooperative
    // launch each (grid.sync between passes). 8 launches -> 3, descales inline.
    fcfa_launch_stage(Q->data, q_fp8.get(), d, n_q, h_q,
        (int64_t) Q->nb[1], (int64_t) Q->nb[2], /*f32*/1, amax.get()+0, dsb.get()+0, /*Q*/1, dsb.get(), stream);
    fcfa_launch_stage(base_k, k_fp8.get(), d, dim_kv_fp8, h_k,
        k_nb1, k_nb2, /*f16*/0, amax.get()+1, dsb.get()+1, /*K*/0, dsb.get(), stream);
    fcfa_launch_stage(base_v, v_fp8.get(), d, dim_kv_fp8, h_k,
        v_nb1, v_nb2, /*f16*/0, amax.get()+2, dsb.get()+2, /*V*/2, dsb.get(), stream);

    std::unordered_map<int64_t, void *> pack = {
        {UID_Q,      q_fp8.get()},
        {UID_K,      k_fp8.get()},
        {UID_V,      v_fp8.get()},
        {UID_O,      o_fp8.get()},
        {UID_DQ,     dsb.get() + 0},
        {UID_DK,     dsb.get() + 1},
        {UID_DV,     dsb.get() + 2},
        {UID_DS,     dsb.get() + 3},
        {UID_SS,     dsb.get() + 4},
        {UID_SO,     dsb.get() + 5},
        {UID_AMAX_S, amo.get() + 0},
        {UID_AMAX_O, amo.get() + 1},
    };
    auto est = ent.graph->execute(st.handle, pack, ent.ws_size > 0 ? (void *) ws.get() : nullptr);
    if (!est.is_good()) {
        if (fcfa_log_enabled()) {
            fprintf(stderr, "[FCFA-cuDNN-FP8] execute FAILED: %s — falling back\n", est.get_message().c_str());
        }
        return false;
    }
    {
        const int64_t n = dv*n_q*h_q;
        const int blk = 256;
        // dequant O: multiply by 1/scale_o = amax_v/448 = dsb[2] (descale_v)
        k_fcfa_o_fp8_to_f32<<<(unsigned) ((n + blk - 1)/blk), blk, 0, stream>>>(
            o_fp8.get(), (float *) dst->data, n, dsb.get() + 2);
    }
    return true;
}

bool ggml_cuda_flash_attn_ext_cudnn_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool disabled = getenv("GGML_CUDA_NO_CUDNN_FA") != nullptr;
    if (disabled) {
        return false;
    }
    static const int64_t min_q = [] {
        const char * s = getenv("GGML_CUDA_CUDNN_FA_MIN_Q");
        return s ? atoll(s) : 256;
    }();

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float scale, max_bias, logit_softcap;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int64_t d   = Q->ne[0];
    const int64_t n_q = Q->ne[1];
    const int64_t h_q = Q->ne[2];
    const int64_t h_k = K->ne[2];
    const int64_t n_kv = K->ne[1];
    const int64_t dv  = V->ne[0];

    // static eligibility gates
    if (mask == nullptr || sinks != nullptr || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (n_q < min_q) { // prefill only; decode keeps the tuned builtin path
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16) {
        return false;
    }
    if (d != 128 || dv != 128) { // the only shape validated so far
        return false;
    }
    if (h_k <= 0 || h_q % h_k != 0) {
        return false;
    }
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[3] != 1 || mask->ne[2] != 1) {
        return false;
    }
    if (mask->ne[0] < n_kv || mask->ne[1] < n_q) {
        return false;
    }
    // element strides (all tensors f16/f32 with d contiguous required)
    if (Q->nb[0] != sizeof(float) || K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half)) {
        return false;
    }
    if (!ggml_is_contiguous(mask) || !ggml_is_contiguous(dst)) {
        return false;
    }

    fcfa_state & st = fcfa_get_state();
    cudaStream_t stream = ctx.stream();

    // ---- mask band proof (cached per graph eval) ----
    const uint64_t gen = g_fcfa_gen.load(std::memory_order_relaxed);
    if (!(st.mask_checked && st.mask_gen == gen && st.mask_ptr == mask->data &&
          st.mask_n_q == n_q && st.mask_n_kv == n_kv)) {
        // Probe on a DEDICATED side stream: the mask is a graph INPUT uploaded
        // by the host before any kernel of this eval runs, so the probe needs
        // none of the queued main-stream work. Syncing the main stream here
        // instead was measured to stall the GPU ~4ms per graph eval (the whole
        // queued pipeline had to drain before the verdict came back).
        static cudaStream_t probe_stream = [] {
            cudaStream_t s = nullptr;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            return s;
        }();
        // dedicated buffer (NOT ctx.pool(): pool blocks are stream-ordered on the
        // main stream; writing one from the side stream could race in-flight users)
        static int32_t * probe_buf = nullptr;
        static int64_t   probe_cap = 0;
        if (3*n_q > probe_cap) {
            if (probe_buf) cudaFree(probe_buf);
            CUDA_CHECK(cudaMalloc((void **) &probe_buf, 3*n_q*sizeof(int32_t)));
            probe_cap = 3*n_q;
        }
        const int64_t row_stride = mask->nb[1]/sizeof(half);
        k_fcfa_mask_probe<<<n_q, 256, 0, probe_stream>>>(
            (const half *) mask->data, n_kv, row_stride, probe_buf, probe_buf + n_q, probe_buf + 2*n_q);
        st.h_probe.resize(3*n_q);
        CUDA_CHECK(cudaMemcpyAsync(st.h_probe.data(), probe_buf, 3*n_q*sizeof(int32_t), cudaMemcpyDeviceToHost, probe_stream));
        CUDA_CHECK(cudaStreamSynchronize(probe_stream));

        const int32_t * lo  = st.h_probe.data();
        const int32_t * hi  = st.h_probe.data() + n_q;
        const int32_t * cnt = st.h_probe.data() + 2*n_q;
        // Accept a causal band at constant offset L (kv-unified slot base):
        // row i allows exactly the contiguous run [L, L + D + i].
        const int32_t L = lo[0];
        const int32_t D = hi[0] - L;
        bool ok = hi[0] >= 0 && D >= 0;
        for (int64_t i = 0; ok && i < n_q; i++) {
            ok = lo[i] == L && hi[i] == L + D + (int32_t) i && cnt[i] == hi[i] - lo[i] + 1;
        }
        st.mask_gen     = gen;
        st.mask_ptr     = mask->data;
        st.mask_n_q     = n_q;
        st.mask_n_kv    = n_kv;
        st.mask_checked = true;
        st.mask_is_band = ok;
        st.band_lo      = ok ? L : 0;
        st.band_n_past  = ok ? D : 0;
        if (fcfa_log_enabled()) {
            fprintf(stderr, "[FCFA-cuDNN] mask probe n_q=%ld n_kv=%ld -> %s (lo=%d n_past=%d)\n",
                (long) n_q, (long) n_kv, ok ? "CAUSAL-BAND" : "not a band, fallback", (int) L, (int) D);
        }
    }
    if (!st.mask_is_band) {
        return false;
    }
    const int32_t band_lo = st.band_lo;
    const int32_t n_past  = st.band_n_past;
    const int32_t seq_kv  = n_past + (int32_t) n_q; // real keys from band_lo; rows beyond are padding-masked

    // ---- graph lookup / build ----
    if (st.handle == nullptr) {
        if (cudnnCreate(&st.handle) != CUDNN_STATUS_SUCCESS) {
            if (fcfa_log_enabled()) {
                fprintf(stderr, "[FCFA-cuDNN] cudnnCreate failed, disabling\n");
            }
            st.mask_is_band = false; // avoid retry storm this eval
            return false;
        }
    }
    cudnnSetStream(st.handle, stream);

    // K/V window: skip band_lo rows (kv-unified slot base) and bucket the row
    // count to keep the graph cache small — rows past seq_kv are padding-masked
    // (proven inert in prototyping), but must stay inside the KV view.
    const int64_t kv_rem = n_kv - band_lo;
    if ((int64_t) seq_kv > kv_rem) {
        return false; // malformed band (shouldn't happen)
    }
    static const int64_t kv_bucket = [] {
        const char * s = getenv("GGML_CUDA_CUDNN_FA_KV_BUCKET");
        const int64_t v = s ? atoll(s) : 512;
        return v > 0 ? v : 512;
    }();
    int64_t dim_kv = ((int64_t) seq_kv + kv_bucket - 1) & ~(kv_bucket - 1);
    if (dim_kv > kv_rem) {
        dim_kv = kv_rem;
    }

    // FP8 (E4M3) path — feature-flagged, isolated cache, A/B against HALF baseline.
    // Adaptive gate: FP8 staging (3x amax + 3x quantize + O dequant per call) only
    // pays off once attention's quadratic term dominates. Measured crossover ~4k
    // (pp2048 FP8 slower, pp8192 FP8 +6.3% NVFP4 / +4.4% MXFP4). Below the
    // threshold, stay on the validated HALF path so FP8 is never a regression.
    // FP8 (E4M3) attention default-ON: validated +9.1% NVFP4 prefill at pp8192, needle
    // retrieval correct at 21k-token context. Gated to seq_kv>=fcfa_fp8_min_kv (4096) so
    // short-context and decode stay byte-identical on the HALF path, and any FP8
    // build/execute failure falls through to HALF below. Escape hatch: GGML_CUDA_CUDNN_FA_FP8_OFF.
    static const bool fcfa_use_fp8 = getenv("GGML_CUDA_CUDNN_FA_FP8_OFF") == nullptr;
    static const int64_t fcfa_fp8_min_kv = [] {
        const char * s = getenv("GGML_CUDA_CUDNN_FA_FP8_MIN_KV");
        return s ? atoll(s) : 4096;
    }();
    if (fcfa_use_fp8 && (int64_t) seq_kv >= fcfa_fp8_min_kv) {
        if (fcfa_run_fp8(ctx, st, dst, Q, K, V, band_lo, seq_kv, dim_kv,
                         scale, n_q, h_q, h_k, d, dv, stream)) {
            return true;
        }
        // fall through to HALF path on any FP8 failure (build/execute) so
        // correctness/perf is never worse than the validated baseline.
    }

    fcfa_graph_key key = {};
    key.n_q  = n_q;  key.n_kv = dim_kv;
    key.h_q  = h_q;  key.h_k  = h_k;
    key.d    = d;    key.dv   = dv;
    key.q_nb1 = 0;   key.q_nb2 = 0; // Q staged contiguous
    key.k_nb1 = (int64_t) (K->nb[1]/sizeof(half));
    key.k_nb2 = (int64_t) (K->nb[2]/sizeof(half));
    key.v_nb1 = (int64_t) (V->nb[1]/sizeof(half));
    key.v_nb2 = (int64_t) (V->nb[2]/sizeof(half));
    memcpy(&key.scale_bits, &scale, sizeof(scale));

    auto it = st.graphs->find(key);
    if (it == st.graphs->end()) {
        fcfa_graph_entry ent;
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);
        // Q staged: contiguous interleaved [s][h][d]
        auto Qt = graph->tensor(fe::graph::Tensor_attributes().set_uid(UID_Q)
            .set_dim({1, h_q, n_q, d}).set_stride({d*n_q*h_q, d, d*h_q, 1}));
        auto Kt = graph->tensor(fe::graph::Tensor_attributes().set_uid(UID_K)
            .set_dim({1, h_k, dim_kv, d}).set_stride({d*dim_kv*h_k, key.k_nb2, key.k_nb1, 1}));
        auto Vt = graph->tensor(fe::graph::Tensor_attributes().set_uid(UID_V)
            .set_dim({1, h_k, dim_kv, d}).set_stride({d*dim_kv*h_k, key.v_nb2, key.v_nb1, 1}));
        auto seq_q_t  = graph->tensor(fe::graph::Tensor_attributes().set_uid(UID_SEQ_Q)
            .set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::INT32));
        auto seq_kv_t = graph->tensor(fe::graph::Tensor_attributes().set_uid(UID_SEQ_KV)
            .set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::INT32));

        auto opts = fe::graph::SDPA_attributes()
            .set_generate_stats(false)
            .set_attn_scale(scale)
            .set_diagonal_alignment(fe::DiagonalAlignment_t::BOTTOM_RIGHT)
            .set_diagonal_band_right_bound(0)
            .set_padding_mask(true).set_seq_len_q(seq_q_t).set_seq_len_kv(seq_kv_t);

        auto [Ot, stats] = graph->sdpa(Qt, Kt, Vt, opts);
        GGML_UNUSED(stats);
        // O staged f16, layout identical to ggml f32 dst: interleaved [s][h][d]
        Ot->set_output(true).set_uid(UID_O)
          .set_dim({1, h_q, n_q, dv}).set_stride({dv*n_q*h_q, dv, dv*h_q, 1});

        auto bst = graph->build(st.handle, {fe::HeurMode_t::A});
        if (!bst.is_good()) {
            ent.build_failed = true;
            if (fcfa_log_enabled()) {
                fprintf(stderr, "[FCFA-cuDNN] graph build FAILED (n_q=%ld dim_kv=%ld): %s\n",
                    (long) n_q, (long) dim_kv, bst.get_message().c_str());
            }
        } else {
            ent.graph = graph;
            (void) graph->get_workspace_size(ent.ws_size);
            if (fcfa_log_enabled()) {
                fprintf(stderr, "[FCFA-cuDNN] graph built n_q=%ld dim_kv=%ld ws=%.1f MB (cache size %zu)\n",
                    (long) n_q, (long) dim_kv, ent.ws_size/1048576.0, st.graphs->size() + 1);
            }
        }
        it = st.graphs->emplace(key, std::move(ent)).first;
    }
    if (it->second.build_failed) {
        return false;
    }
    const fcfa_graph_entry & ent = it->second;

    // ---- staging + execute ----
    ggml_cuda_pool_alloc<half>    q_f16(ctx.pool(), d*n_q*h_q);
    ggml_cuda_pool_alloc<half>    o_f16(ctx.pool(), dv*n_q*h_q);
    ggml_cuda_pool_alloc<int32_t> seqs (ctx.pool(), 2);
    ggml_cuda_pool_alloc<char>    ws;
    if (ent.ws_size > 0) {
        ws.alloc(ctx.pool(), ent.ws_size);
    }

    {
        const int64_t n = d*n_q*h_q;
        const int blk = 256;
        k_fcfa_q_f32_to_f16<<<(unsigned) ((n + blk - 1)/blk), blk, 0, stream>>>(
            (const char *) Q->data, q_f16.get(), d, n_q, h_q, (int64_t) Q->nb[1], (int64_t) Q->nb[2]);
    }
    k_fcfa_set_seq<<<1, 1, 0, stream>>>(seqs.get(), (int32_t) n_q, seq_kv);

    std::unordered_map<int64_t, void *> pack = {
        {UID_Q,      q_f16.get()},
        {UID_K,      (char *) K->data + (int64_t) band_lo*K->nb[1]},
        {UID_V,      (char *) V->data + (int64_t) band_lo*V->nb[1]},
        {UID_O,      o_f16.get()},
        {UID_SEQ_Q,  seqs.get()},
        {UID_SEQ_KV, seqs.get() + 1},
    };
    auto est = ent.graph->execute(st.handle, pack, ent.ws_size > 0 ? (void *) ws.get() : nullptr);
    if (!est.is_good()) {
        if (fcfa_log_enabled()) {
            fprintf(stderr, "[FCFA-cuDNN] execute FAILED: %s — falling back\n", est.get_message().c_str());
        }
        return false; // builtin kernel will overwrite dst; o_f16 scratch is discarded
    }

    {
        const int64_t n = dv*n_q*h_q;
        const int blk = 256;
        k_fcfa_o_f16_to_f32<<<(unsigned) ((n + blk - 1)/blk), blk, 0, stream>>>(o_f16.get(), (float *) dst->data, n);
    }
    return true;
}

#endif // GGML_CUDA_USE_CUDNN
