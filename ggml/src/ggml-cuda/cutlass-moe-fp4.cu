// CUTLASS grouped NVFP4 MoE bridge — see cutlass-moe-fp4.cuh.
// Reference impl (proven): .frankencoder/engine/fp4moe-port/validate_glue.cu.
#include "cutlass-moe-fp4.cuh"
#include "cutlass-moe-fp4-glue.cuh"

#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

using namespace cute;

// ===================== grouped NVFP4 -> bf16 type config (matches validated kernel) =====================
using ElementInput   = cutlass::float_e2m1_t;
using ElementA       = cutlass::nv_float4_t<ElementInput>;
using LayoutATag     = cutlass::layout::RowMajor;
static constexpr int AlignmentA = 32;
using ElementB       = cutlass::nv_float4_t<ElementInput>;
using LayoutBTag     = cutlass::layout::ColumnMajor;
static constexpr int AlignmentB = 32;
using ElementD       = cutlass::bfloat16_t;
using ElementC       = cutlass::bfloat16_t;
using LayoutCTag     = cutlass::layout::RowMajor;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag        = cutlass::arch::Sm120;
using OperatorClass  = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128,_128,_128>;
using ClusterShape   = Shape<_1,_1,_1>;
using ProblemShape   = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag *, AlignmentC,
    ElementD, LayoutCTag *, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag *, AlignmentA,
    ElementB, LayoutBTag *, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
using ElementAData = typename ElementA::DataType;
using ElementBData = typename ElementB::DataType;

#define QSUB 16

// ===================== persistent weight-repack cache =====================
struct WeightCache { void* dB=nullptr; void* dSFB=nullptr; int E=0,N=0,K=0; };
static std::mutex g_wc_mtx;
static std::unordered_map<const void*, WeightCache> g_wc;

static const WeightCache& get_repacked_weights(
        const void* w_blocks, int E, int N, int K, size_t nb01, size_t nb02, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_wc_mtx);
    auto it = g_wc.find(w_blocks);
    if (it != g_wc.end()) return it->second;
    WeightCache wc; wc.E=E; wc.N=N; wc.K=K;
    size_t Bsz   = (size_t)E * N * (K/2);
    size_t SFBsz = (size_t)E * N * (K/QSUB);
    cudaMalloc(&wc.dB, Bsz);
    cudaMalloc(&wc.dSFB, SFBsz);
    cudaMemsetAsync(wc.dSFB, 0, SFBsz, stream);
    ggml_cuda_nvfp4_repack_weights(w_blocks, wc.dB, wc.dSFB, E, N, K, nb01, nb02, stream);
    cudaStreamSynchronize(stream); // repack once; make weights visible before first GEMM
    auto res = g_wc.emplace(w_blocks, wc);
    return res.first->second;
}

template<class V> static void* upload(const V& v, cudaStream_t s){
    using T = typename V::value_type;
    void* d=nullptr; cudaMallocAsync(&d, v.size()*sizeof(T), s);
    cudaMemcpyAsync(d, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice, s);
    return d;
}

extern "C" int ggml_cuda_cutlass_moe_nvfp4_supported(int E, int N, int K){
    if (E <= 0 || N <= 0 || K <= 0) return 0;
    if (N % 128 != 0) return 0;      // SFB swizzle atom = 128 rows
    if (K % 64  != 0) return 0;      // NVFP4 block = 64
    return 1;
}

extern "C" int ggml_cuda_cutlass_moe_nvfp4(
        const void * w_blocks, const float * src1_sorted, float * dst_sorted,
        const int32_t * tokens_per_expert, int E, int N, int K,
        size_t nb01, size_t nb02, cudaStream_t stream) {
    if (!ggml_cuda_cutlass_moe_nvfp4_supported(E, N, K)) return 1;

    // row/expert bookkeeping (host)
    std::vector<int32_t> expert_off(E), sf_off(E);
    int Mtot = 0, sf_rows = 0;
    std::vector<int> active;              // expert ids with tokens>0
    for (int e = 0; e < E; ++e) {
        expert_off[e] = Mtot;
        sf_off[e]     = sf_rows;
        int me = tokens_per_expert[e];
        if (me > 0) active.push_back(e);
        Mtot    += me;
        sf_rows += ((me + 127) / 128) * 128;
    }
    if (Mtot == 0 || active.empty()) return 1;

    std::vector<int32_t> row_expert(Mtot);
    for (int e = 0; e < E; ++e)
        for (int i = 0; i < tokens_per_expert[e]; ++i)
            row_expert[expert_off[e] + i] = e;

    const WeightCache& wc = get_repacked_weights(w_blocks, E, N, K, nb01, nb02, stream);

    // ---- activation quant (glue) ----
    float* dg=nullptr; cudaMallocAsync(&dg, (size_t)Mtot*sizeof(float), stream);
    ggml_cuda_nvfp4_act_row_scale(src1_sorted, dg, Mtot, K, stream);

    void *dAq=nullptr, *dSFA=nullptr;
    cudaMallocAsync(&dAq, (size_t)Mtot*(K/2), stream);
    cudaMallocAsync(&dSFA, (size_t)sf_rows*(K/QSUB), stream);
    cudaMemsetAsync(dSFA, 0, (size_t)sf_rows*(K/QSUB), stream);

    void *dRE=nullptr,*dEO=nullptr,*dSO=nullptr;
    cudaMallocAsync(&dRE, (size_t)Mtot*4, stream);
    cudaMallocAsync(&dEO, (size_t)E*4, stream);
    cudaMallocAsync(&dSO, (size_t)E*4, stream);
    cudaMemcpyAsync(dRE, row_expert.data(), (size_t)Mtot*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dEO, expert_off.data(), (size_t)E*4, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dSO, sf_off.data(),     (size_t)E*4, cudaMemcpyHostToDevice, stream);
    ggml_cuda_nvfp4_quant_acts(src1_sorted, dg, dAq, dSFA,
        (const int32_t*)dRE, (const int32_t*)dEO, (const int32_t*)dSO, Mtot, K, stream);

    // ---- bf16 output buffer ----
    ElementD* dD=nullptr; cudaMallocAsync(&dD, (size_t)Mtot*N*sizeof(ElementD), stream);

    // ---- per-group arrays (active experts only) ----
    const int G = (int)active.size();
    std::vector<typename ProblemShape::UnderlyingProblemShape> ps(G);
    std::vector<const ElementAData*> hA(G); std::vector<const ElementBData*> hB(G);
    std::vector<const ElementC*> hC(G); std::vector<ElementD*> hD(G);
    std::vector<const ElementSF*> hSFA(G), hSFB(G);
    std::vector<StrideA> hSA(G); std::vector<StrideB> hSB(G);
    std::vector<StrideC> hSC(G); std::vector<StrideD> hSD(G);
    std::vector<LayoutSFA> hLA(G); std::vector<LayoutSFB> hLB(G);

    for (int g = 0; g < G; ++g) {
        int e  = active[g];
        int Me = tokens_per_expert[e];
        ps[g] = {Me, N, K};
        hA[g]  = reinterpret_cast<const ElementAData*>((const uint8_t*)dAq + (size_t)expert_off[e]*(K/2));
        hB[g]  = reinterpret_cast<const ElementBData*>((const uint8_t*)wc.dB + (size_t)e*N*(K/2));
        hC[g]  = nullptr;
        hD[g]  = dD + (size_t)expert_off[e]*N;
        hSFA[g]= reinterpret_cast<const ElementSF*>((const uint8_t*)dSFA + (size_t)sf_off[e]*(K/QSUB));
        hSFB[g]= reinterpret_cast<const ElementSF*>((const uint8_t*)wc.dSFB + (size_t)e*N*(K/QSUB));
        hSA[g] = cutlass::make_cute_packed_stride(StrideA{}, {Me, K, 1});
        hSB[g] = cutlass::make_cute_packed_stride(StrideB{}, {N,  K, 1});
        hSC[g] = cutlass::make_cute_packed_stride(StrideC{}, {Me, N, 1});
        hSD[g] = cutlass::make_cute_packed_stride(StrideD{}, {Me, N, 1});
        hLA[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(Me, N, K, 1));
        hLB[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(Me, N, K, 1));
    }

    auto* pA=(const ElementAData**)upload(hA,stream);
    auto* pB=(const ElementBData**)upload(hB,stream);
    auto* pC=(const ElementC**)upload(hC,stream);
    auto* pD=(ElementD**)upload(hD,stream);
    auto* pSFA=(const ElementSF**)upload(hSFA,stream);
    auto* pSFB=(const ElementSF**)upload(hSFB,stream);
    auto* pSA=(StrideA*)upload(hSA,stream);
    auto* pSB=(StrideB*)upload(hSB,stream);
    auto* pSC=(StrideC*)upload(hSC,stream);
    auto* pSD=(StrideD*)upload(hSD,stream);
    auto* pLA=(LayoutSFA*)upload(hLA,stream);
    auto* pLB=(LayoutSFB*)upload(hLB,stream);
    auto* pPS=(typename ProblemShape::UnderlyingProblemShape*)upload(ps,stream);
    cudaStreamSynchronize(stream); // host arrays consumed; ptr arrays live on device

    cutlass::KernelHardwareInfo hw; hw.device_id = 0;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    typename Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
    args.problem_shape = {G, pPS, ps.data()};
    args.mainloop = { pA, pSA, pB, pSB, pSFA, pLA, pSFB, pLB };
    decltype(args.epilogue.thread) fusion; fusion.alpha = 1.0f; fusion.beta = 0.0f;
    args.epilogue = { fusion, pC, pSC, pD, pSD };
    args.hw_info = hw;

    Gemm gemm;
    size_t wssz = Gemm::get_workspace_size(args);
    void* wsbuf=nullptr; cudaMallocAsync(&wsbuf, wssz ? wssz : 1, stream);
    int rc = 0;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) rc = 2;
    else if (gemm.initialize(args, wsbuf) != cutlass::Status::kSuccess) rc = 3;
    else if (gemm.run(stream) != cutlass::Status::kSuccess) rc = 4;

    if (rc == 0) {
        // dst_sorted[m,n] = bf16(D[m,n]) * g[m]
        ggml_cuda_bf16_to_f32_rowscaled(dD, dst_sorted, dg, Mtot, N, stream);
    }

    // free per-call temporaries (stream-ordered)
    for (void* p : {(void*)pA,(void*)pB,(void*)pC,(void*)pD,(void*)pSFA,(void*)pSFB,
                    (void*)pSA,(void*)pSB,(void*)pSC,(void*)pSD,(void*)pLA,(void*)pLB,(void*)pPS,
                    (void*)dAq,(void*)dSFA,(void*)dRE,(void*)dEO,(void*)dSO,(void*)dg,(void*)dD,wsbuf})
        cudaFreeAsync(p, stream);

    return rc;
}

// ============================================================================
// ZERO-SYNC prefill path (2026-07-02)
// All group arguments built ON DEVICE from expert_bounds; host_problem_shapes
// = nullptr (same technique as vLLM's cutlass_moe_mm). No host syncs, no
// per-call cudaMalloc: persistent grow-only scratch. Empty experts get Me=0
// problem shapes, which the group scheduler skips.
// ============================================================================

// one thread per expert: fill grouped-GEMM argument arrays from device bounds
__global__ void k_build_group_args(
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ sf_offsets,
        const uint8_t * wB, const uint8_t * wSFB,
        const uint8_t * dAq, const uint8_t * dSFA, ElementD * dD,
        typename ProblemShape::UnderlyingProblemShape * ps,
        const ElementAData ** pA, const ElementBData ** pB,
        const ElementC ** pC, ElementD ** pD,
        const ElementSF ** pSFA, const ElementSF ** pSFB,
        StrideA * sa, StrideB * sb, StrideC * sc, StrideD * sd,
        LayoutSFA * la, LayoutSFB * lb,
        int E, int N, int K) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const int off = expert_bounds[e];
    const int Me  = expert_bounds[e + 1] - off;
    ps[e]   = {Me, N, K};
    pA[e]   = reinterpret_cast<const ElementAData *>(dAq + (size_t) off * (K / 2));
    pB[e]   = reinterpret_cast<const ElementBData *>(wB + (size_t) e * N * (K / 2));
    pC[e]   = nullptr;
    pD[e]   = dD + (size_t) off * N;
    pSFA[e] = reinterpret_cast<const ElementSF *>(dSFA + (size_t) sf_offsets[e] * (K / QSUB));
    pSFB[e] = reinterpret_cast<const ElementSF *>(wSFB + (size_t) e * N * (K / QSUB));
    sa[e]   = cutlass::make_cute_packed_stride(StrideA{}, {Me, K, 1});
    sb[e]   = cutlass::make_cute_packed_stride(StrideB{}, {N,  K, 1});
    sc[e]   = cutlass::make_cute_packed_stride(StrideC{}, {Me, N, 1});
    sd[e]   = cutlass::make_cute_packed_stride(StrideD{}, {Me, N, 1});
    la[e]   = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(Me, N, K, 1));
    lb[e]   = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(Me, N, K, 1));
}

// persistent grow-only scratch (single device assumed; guarded by mutex)
struct PrefillScratch {
    // per-expert argument arrays (sized E; realloc only if E grows)
    void * args_blob = nullptr;   // single allocation carved into the arrays below
    int    E_cap     = 0;
    typename ProblemShape::UnderlyingProblemShape * ps = nullptr;
    const ElementAData ** pA = nullptr; const ElementBData ** pB = nullptr;
    const ElementC ** pC = nullptr; ElementD ** pD = nullptr;
    const ElementSF ** pSFA = nullptr; const ElementSF ** pSFB = nullptr;
    StrideA * sa = nullptr; StrideB * sb = nullptr;
    StrideC * sc = nullptr; StrideD * sd = nullptr;
    LayoutSFA * la = nullptr; LayoutSFB * lb = nullptr;
    int32_t * sf_off = nullptr;   // [E]
    // token-dependent buffers (grow-only)
    uint8_t  * dAq  = nullptr; size_t aq_cap  = 0;   // Mtot*(K/2)
    uint8_t  * dSFA = nullptr; size_t sfa_cap = 0;   // sf_rows*(K/QSUB)
    float    * dG   = nullptr; size_t g_cap   = 0;   // Mtot
    ElementD * dD   = nullptr; size_t d_cap   = 0;   // Mtot*N
    void     * ws   = nullptr; size_t ws_cap  = 0;   // GEMM workspace
};
static std::mutex g_pf_mtx;
static PrefillScratch g_pf;

static void * grow(void ** p, size_t * cap, size_t need) {
    if (need > *cap) {
        if (*p) cudaFree(*p);
        size_t sz = need + need / 4;            // 25% headroom to damp regrowth
        cudaMalloc(p, sz);
        *cap = sz;
    }
    return *p;
}

extern "C" int ggml_cuda_cutlass_moe_nvfp4_prefill(
        const void * w_blocks, const float * src1, const int32_t * ids_src1,
        const int32_t * ids_dst, const int32_t * expert_bounds, float * dst,
        int E, int N, int K, int Mtot, int64_t s11, int64_t s1,
        size_t nb01, size_t nb02, cudaStream_t stream) {
    if (!ggml_cuda_cutlass_moe_nvfp4_supported(E, N, K)) return 1;
    if (Mtot <= 0 || E > 1024) return 1;
    if ((size_t) K * sizeof(float) + 128 > 96 * 1024) return 1; // gather kernel smem cap (sm120: 99KB)

    std::lock_guard<std::mutex> lk(g_pf_mtx);
    const WeightCache & wc = get_repacked_weights(w_blocks, E, N, K, nb01, nb02, stream);

    // ---- scratch (grow-only; steady-state = zero allocations) ----
    if (E > g_pf.E_cap) {
        if (g_pf.args_blob) cudaFree(g_pf.args_blob);
        const size_t per_e =
            sizeof(typename ProblemShape::UnderlyingProblemShape) +
            2 * sizeof(void *) + 2 * sizeof(void *) + 2 * sizeof(void *) +
            sizeof(StrideA) + sizeof(StrideB) + sizeof(StrideC) + sizeof(StrideD) +
            sizeof(LayoutSFA) + sizeof(LayoutSFB) + sizeof(int32_t);
        cudaMalloc(&g_pf.args_blob, (per_e + 64) * (size_t) E); // slack for 16B alignment carving
        uint8_t * base = (uint8_t *) g_pf.args_blob;
        auto carve = [&](size_t bytes) { void * r = base; base += (bytes + 15) & ~size_t(15); return r; };
        g_pf.ps   = (typename ProblemShape::UnderlyingProblemShape *) carve(sizeof(*g_pf.ps) * E);
        g_pf.pA   = (const ElementAData **) carve(sizeof(void *) * E);
        g_pf.pB   = (const ElementBData **) carve(sizeof(void *) * E);
        g_pf.pC   = (const ElementC **)     carve(sizeof(void *) * E);
        g_pf.pD   = (ElementD **)           carve(sizeof(void *) * E);
        g_pf.pSFA = (const ElementSF **)    carve(sizeof(void *) * E);
        g_pf.pSFB = (const ElementSF **)    carve(sizeof(void *) * E);
        g_pf.sa   = (StrideA *)   carve(sizeof(StrideA) * E);
        g_pf.sb   = (StrideB *)   carve(sizeof(StrideB) * E);
        g_pf.sc   = (StrideC *)   carve(sizeof(StrideC) * E);
        g_pf.sd   = (StrideD *)   carve(sizeof(StrideD) * E);
        g_pf.la   = (LayoutSFA *) carve(sizeof(LayoutSFA) * E);
        g_pf.lb   = (LayoutSFB *) carve(sizeof(LayoutSFB) * E);
        g_pf.sf_off = (int32_t *) carve(sizeof(int32_t) * E);
        g_pf.E_cap = E;
    }
    const size_t sf_rows = (size_t) Mtot + 128 * (size_t) E;    // worst-case round-up padding
    grow((void **) &g_pf.dAq,  &g_pf.aq_cap,  (size_t) Mtot * (K / 2));
    grow((void **) &g_pf.dSFA, &g_pf.sfa_cap, sf_rows * (K / QSUB));
    grow((void **) &g_pf.dG,   &g_pf.g_cap,   (size_t) Mtot * sizeof(float));
    grow((void **) &g_pf.dD,   &g_pf.d_cap,   (size_t) Mtot * N * sizeof(ElementD));

    // ---- device-side prep: sf_offsets -> gather+quant -> group args (no syncs) ----
    ggml_cuda_nvfp4_sf_offsets(expert_bounds, g_pf.sf_off, E, stream);
    cudaMemsetAsync(g_pf.dSFA, 0, sf_rows * (K / QSUB), stream);
    ggml_cuda_nvfp4_gather_quant(src1, ids_src1, expert_bounds, g_pf.sf_off,
                                 g_pf.dG, g_pf.dAq, g_pf.dSFA, E, Mtot, K, s11, stream);
    {
        const int threads = 128;
        const int blocks = (E + threads - 1) / threads;
        k_build_group_args<<<blocks, threads, 0, stream>>>(
            expert_bounds, g_pf.sf_off, (const uint8_t *) wc.dB, (const uint8_t *) wc.dSFB,
            g_pf.dAq, g_pf.dSFA, g_pf.dD,
            g_pf.ps, g_pf.pA, g_pf.pB, g_pf.pC, g_pf.pD, g_pf.pSFA, g_pf.pSFB,
            g_pf.sa, g_pf.sb, g_pf.sc, g_pf.sd, g_pf.la, g_pf.lb, E, N, K);
    }

    // ---- grouped GEMM, device-side problem shapes (host shapes = nullptr) ----
    static cutlass::KernelHardwareInfo hw = [] {
        cutlass::KernelHardwareInfo h; h.device_id = 0;
        h.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
        return h;
    }();

    typename Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
    args.problem_shape = {E, g_pf.ps, nullptr};
    args.mainloop = { g_pf.pA, g_pf.sa, g_pf.pB, g_pf.sb, g_pf.pSFA, g_pf.la, g_pf.pSFB, g_pf.lb };
    decltype(args.epilogue.thread) fusion; fusion.alpha = 1.0f; fusion.beta = 0.0f;
    args.epilogue = { fusion, g_pf.pC, g_pf.sc, g_pf.pD, g_pf.sd };
    args.hw_info = hw;

    Gemm gemm;
    const size_t wssz = Gemm::get_workspace_size(args);
    grow(&g_pf.ws, &g_pf.ws_cap, wssz ? wssz : 1);
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 2;
    if (gemm.initialize(args, g_pf.ws, stream) != cutlass::Status::kSuccess) return 3;
    if (gemm.run(stream) != cutlass::Status::kSuccess) return 4;

    // ---- fused scatter + rescale + bf16->f32 straight into unsorted dst ----
    ggml_cuda_nvfp4_scatter_rowscaled(g_pf.dD, g_pf.dG, ids_dst, dst, Mtot, N, s1, stream);
    return 0;
}
