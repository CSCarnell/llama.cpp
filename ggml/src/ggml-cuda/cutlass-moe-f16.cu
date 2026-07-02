// Zero-sync grouped F16 MoE prefill bridge (2026-07-02).
// Replaces the per-expert cuBLAS storm (sorted path in ggml-cuda.cu) for
// GGML_TYPE_F16 experts with ONE CUTLASS 2.x grouped GEMM (sm80 tensor-op,
// GroupScheduleMode::kDeviceOnly -> problem sizes live on DEVICE, no host
// syncs). Weights need NO repack: ggml stores each expert [N,K] row-major
// with K contiguous == CUTLASS ColumnMajor B (K x N), ldb = nb01/2.
//   D[m,n] = sum_k A[m,k] * W[n,k],  A = gathered f16 acts, D = f32.
#include "cutlass-moe-fp4.cuh"
#include "cutlass-moe-fp4-glue.cuh"

#include <mutex>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination.h"

using F16GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::half_t, cutlass::layout::RowMajor,    cutlass::ComplexTransform::kNone, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,      // stages
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

using F16GemmGrouped = cutlass::gemm::device::GemmGrouped<F16GemmKernel>;

// one thread per expert: device-built group args
__global__ void k_build_f16_group_args(
        const int32_t * __restrict__ expert_bounds,
        const cutlass::half_t * A, const uint8_t * W, float * D,
        cutlass::gemm::GemmCoord * ps,
        cutlass::half_t ** pA, cutlass::half_t ** pB, float ** pC, float ** pD,
        int64_t * lda, int64_t * ldb, int64_t * ldc, int64_t * ldd,
        int E, int N, int K, size_t nb02, int64_t ldb_elems) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const int off = expert_bounds[e];
    const int Me  = expert_bounds[e + 1] - off;
    ps[e]  = cutlass::gemm::GemmCoord(Me, N, K);
    pA[e]  = const_cast<cutlass::half_t *>(A) + (size_t) off * K;
    pB[e]  = (cutlass::half_t *) (W + (size_t) e * nb02);
    pC[e]  = D + (size_t) off * N;   // beta = 0: unread, but keep valid
    pD[e]  = D + (size_t) off * N;
    lda[e] = K;
    ldb[e] = ldb_elems;
    ldc[e] = N;
    ldd[e] = N;
}

// persistent grow-only scratch
struct F16Scratch {
    void * args_blob = nullptr; int E_cap = 0;
    cutlass::gemm::GemmCoord * ps = nullptr;
    cutlass::half_t ** pA = nullptr; cutlass::half_t ** pB = nullptr;
    float ** pC = nullptr; float ** pD = nullptr;
    int64_t * lda = nullptr; int64_t * ldb = nullptr;
    int64_t * ldc = nullptr; int64_t * ldd = nullptr;
    void * dA = nullptr; size_t a_cap = 0;   // Mtot*K f16
    void * dD = nullptr; size_t d_cap = 0;   // Mtot*N f32
    void * ws = nullptr; size_t ws_cap = 0;
};
static std::mutex g_f16_mtx;
static F16Scratch g_f16;

static void * f16_grow(void ** p, size_t * cap, size_t need) {
    if (need > *cap) {
        if (*p) cudaFree(*p);
        size_t sz = need + need / 4;
        cudaMalloc(p, sz);
        *cap = sz;
    }
    return *p;
}

// routing cache lives in cutlass-moe-fp4.cu
extern "C" void ggml_cuda_moe_routing_get(
    const ggml_cuda_moe_ids_args * a, cudaStream_t stream,
    const int32_t ** ids_src1, const int32_t ** ids_dst,
    const int32_t ** bounds, uint64_t * generation);

extern "C" int ggml_cuda_cutlass_moe_f16_prefill(
        const void * w_f16, const float * src1, const ggml_cuda_moe_ids_args * ids,
        float * dst, int E, int N, int K, int Mtot, int64_t s11, int64_t s1,
        size_t nb01, size_t nb02, const float * fuse_w, cudaStream_t stream) {
    if (E <= 0 || N <= 0 || K <= 0 || Mtot <= 0) return 1;
    if (K % 8 != 0 || N % 4 != 0) return 1;                    // alignment
    if ((nb01 / sizeof(cutlass::half_t)) % 8 != 0) return 1;   // ldb alignment
    if (nb01 % sizeof(cutlass::half_t) != 0) return 1;

    const int32_t * ids_src1; const int32_t * ids_dst; const int32_t * expert_bounds;
    uint64_t routing_gen;
    ggml_cuda_moe_routing_get(ids, stream, &ids_src1, &ids_dst, &expert_bounds, &routing_gen);

    std::lock_guard<std::mutex> lk(g_f16_mtx);

    // ---- scratch ----
    if (E > g_f16.E_cap) {
        if (g_f16.args_blob) cudaFree(g_f16.args_blob);
        const size_t per_e = sizeof(cutlass::gemm::GemmCoord) + 4 * sizeof(void *) + 4 * sizeof(int64_t);
        cudaMalloc(&g_f16.args_blob, (per_e + 64) * (size_t) E);
        uint8_t * base = (uint8_t *) g_f16.args_blob;
        auto carve = [&](size_t bytes) { void * r = base; base += (bytes + 15) & ~size_t(15); return r; };
        g_f16.ps  = (cutlass::gemm::GemmCoord *) carve(sizeof(cutlass::gemm::GemmCoord) * E);
        g_f16.pA  = (cutlass::half_t **) carve(sizeof(void *) * E);
        g_f16.pB  = (cutlass::half_t **) carve(sizeof(void *) * E);
        g_f16.pC  = (float **)  carve(sizeof(void *) * E);
        g_f16.pD  = (float **)  carve(sizeof(void *) * E);
        g_f16.lda = (int64_t *) carve(sizeof(int64_t) * E);
        g_f16.ldb = (int64_t *) carve(sizeof(int64_t) * E);
        g_f16.ldc = (int64_t *) carve(sizeof(int64_t) * E);
        g_f16.ldd = (int64_t *) carve(sizeof(int64_t) * E);
        g_f16.E_cap = E;
    }
    f16_grow(&g_f16.dA, &g_f16.a_cap, (size_t) Mtot * K * sizeof(cutlass::half_t));
    f16_grow(&g_f16.dD, &g_f16.d_cap, (size_t) Mtot * N * sizeof(float));

    // ---- device-side prep: gather+convert, group args (no syncs) ----
    // A-operand reuse across gate->up (same src1 + routing generation)
    static const float * s_a_src1 = nullptr;
    static uint64_t s_a_gen = 0; static int s_a_K = -1, s_a_M = -1;
    const bool a_reusable = s_a_src1 == src1 && s_a_gen == routing_gen &&
                            s_a_K == K && s_a_M == Mtot;
    if (!a_reusable) {
        ggml_cuda_moe_gather_f32_to_f16(src1, ids_src1, g_f16.dA, Mtot, K, s11, stream);
        s_a_src1 = src1; s_a_gen = routing_gen; s_a_K = K; s_a_M = Mtot;
    }
    {
        const int threads = 128;
        const int blocks = (E + threads - 1) / threads;
        k_build_f16_group_args<<<blocks, threads, 0, stream>>>(
            expert_bounds, (const cutlass::half_t *) g_f16.dA, (const uint8_t *) w_f16,
            (float *) g_f16.dD,
            g_f16.ps, g_f16.pA, g_f16.pB, g_f16.pC, g_f16.pD,
            g_f16.lda, g_f16.ldb, g_f16.ldc, g_f16.ldd,
            E, N, K, nb02, (int64_t) (nb01 / sizeof(cutlass::half_t)));
    }

    // ---- ONE grouped GEMM, device-side problem sizes ----
    static const int tb_count = F16GemmGrouped::sufficient(nullptr, E);
    typename F16GemmGrouped::Arguments args(
        g_f16.ps, E, tb_count,
        typename F16GemmGrouped::EpilogueOutputOp::Params(1.0f, 0.0f),
        g_f16.pA, g_f16.pB, g_f16.pC, g_f16.pD,
        g_f16.lda, g_f16.ldb, g_f16.ldc, g_f16.ldd,
        /*host_problem_sizes=*/nullptr);

    F16GemmGrouped gemm;
    const size_t wssz = F16GemmGrouped::get_workspace_size(args);
    f16_grow(&g_f16.ws, &g_f16.ws_cap, wssz ? wssz : 1);
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 2;
    if (gemm.initialize(args, g_f16.ws, stream) != cutlass::Status::kSuccess) return 3;
    if (gemm.run(stream) != cutlass::Status::kSuccess) return 4;

    // ---- scatter f32 rows into unsorted dst ----
    ggml_cuda_moe_scatter_f32((const float *) g_f16.dD, ids_dst, dst, Mtot, N, s1, fuse_w, stream);
    return 0;
}
