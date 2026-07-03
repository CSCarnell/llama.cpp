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

// Mask probe: per q-row, find the lowest/highest allowed KV index and the
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

enum fcfa_uids : int64_t { UID_Q = 1, UID_K, UID_V, UID_O, UID_SEQ_Q = 7, UID_SEQ_KV };

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
    int64_t dim_kv = ((int64_t) seq_kv + 511) & ~511ll;
    if (dim_kv > kv_rem) {
        dim_kv = kv_rem;
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
