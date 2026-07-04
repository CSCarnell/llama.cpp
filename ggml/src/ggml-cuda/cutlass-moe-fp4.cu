// CUTLASS grouped NVFP4 MoE bridge — see cutlass-moe-fp4.cuh.
// Reference impl (proven): .frankencoder/engine/fp4moe-port/validate_glue.cu.
#include "cutlass-moe-fp4.cuh"
#include "cutlass-moe-fp4-glue.cuh"
#include "mmid.cuh"

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
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

using namespace cute;

// ============================================================================
// Format selection. This TU is compiled TWICE: once as-is (NVFP4, nv_float4_t,
// e4m3 SF, block-16) and once via cutlass-moe-mxfp4.cu which does
//   #define FC_FP4_MX 1
//   #include "cutlass-moe-fp4.cu"
// to build the MXFP4 variant (mx_float4_t, e8m0 SF, block-32). All file-scope
// helpers are `static` (internal linkage), so the two object files never collide;
// only the extern "C" entry points differ (FP4_SYM below picks the name).
//   NVFP4: e2m1 codes * ue4m3(sub) * g[row]   (2-level, block-16, numKTiles=K/64)
//   MXFP4: e2m1 codes * ue8m0(block)          (1-level, block-32, numKTiles=K/128)
// The glue kernels, SF layout (Sm1xxBlkScaledConfig::tile_atom_to_shape_SF*), and
// ElementSF type are all derived from the FP4_ELEM type + QSUB, so the body is shared.
// ============================================================================
#ifdef FC_FP4_MX
  #define FP4_ELEM             cutlass::mx_float4_t<ElementInput>
  #define QSUB_VAL             32
  #define FP4_SUPPORTED_KMULT  128            // MXFP4 SF atom needs K % 128 == 0
  // Preprocessor-rename every extern "C" entry + format-specific glue call from
  // its nvfp4 spelling to the mxfp4 one. This rewrites BOTH the definitions in
  // this TU and every internal call site in one place, so the shared body below
  // stays untouched. Format-INDEPENDENT glue (sf_offsets, scatter_rowscaled,
  // gather_accum_rowscaled, act_row_scale, quant_acts, moe_routing_*) is NOT
  // listed here and keeps its nvfp4 symbol (same code serves both builds).
  #define ggml_cuda_cutlass_moe_nvfp4_ffn        ggml_cuda_cutlass_moe_mxfp4_ffn
  #define ggml_cuda_cutlass_moe_nvfp4_prefill    ggml_cuda_cutlass_moe_mxfp4_prefill
  #define ggml_cuda_cutlass_moe_nvfp4_supported  ggml_cuda_cutlass_moe_mxfp4_supported
  #define ggml_cuda_cutlass_dense_nvfp4          ggml_cuda_cutlass_dense_mxfp4
  #define ggml_cuda_cutlass_moe_nvfp4            ggml_cuda_cutlass_moe_mxfp4
  #define ggml_cuda_nvfp4_gather_quant           ggml_cuda_mxfp4_gather_quant
  #define ggml_cuda_nvfp4_swiglu_quant           ggml_cuda_mxfp4_swiglu_quant
  #define ggml_cuda_nvfp4_repack_weights         ggml_cuda_mxfp4_repack_weights
#else
  #define FP4_ELEM             cutlass::nv_float4_t<ElementInput>
  #define QSUB_VAL             16
  #define FP4_SUPPORTED_KMULT  64             // NVFP4 block = 64
#endif

// ===================== grouped FP4 -> bf16 type config (matches validated kernel) =====================
using ElementInput   = cutlass::float_e2m1_t;
using ElementA       = FP4_ELEM;
using LayoutATag     = cutlass::layout::RowMajor;
static constexpr int AlignmentA = 32;
using ElementB       = FP4_ELEM;
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
using ThreadBlockShape = Shape<_128,_128,_128>;  // optimal: swept K256 M256 N256 M64(no-compile) N64 all slower (2026-07-02)
using ClusterShape   = Shape<_1,_1,_1>;  // Sm120 forbids multicast (static_assert): cluster MUST be 1
using ProblemShape   = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

// ---- MoE grouped-GEMM kernel SCHEDULE A/B switch (2026-07-03) ----
// MEASURED RESULT (NVFP4 pp8192 -ub4096 -r3): Pingpong = 28,329 t/s vs
// Cooperative baseline 28,362 t/s -> FLAT (within noise). Confirmed ENGAGED on
// the hot whole-FFN path via FCMOE_DEBUG "WHOLE-FFN fusion ACTIVE" (rc==0, i.e.
// can_implement succeeded on the PtrArray Pingpong-scheduled grouped GEMM).
// Conclusion: 2x-tiles-in-flight does NOT lift throughput here -> the grouped
// MoE GEMM is not tile-count/occupancy limited in a way Pingpong addresses.
// Default reverted to Cooperative (validated baseline). Set FC_MOE_PINGPONG=1
// to re-test. tile-N=128>=16 satisfies the PtrArray Pingpong guard.
#ifndef FC_MOE_PINGPONG
#define FC_MOE_PINGPONG 0
#endif
#if FC_MOE_PINGPONG
using MoEKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
#else
using MoEKernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
#endif


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
    MoEKernelSchedule>::CollectiveOp;

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

// ===================== dense-proj FUSED epilogue (float D + per-row scale EVT) =====================
// The dense Q/K/V/O projection GEMM (E=1 group) normally writes bf16 to a scratch
// buffer, then a standalone k_bf16_to_f32_rowscaled kernel reads it back, multiplies
// by the per-row activation global scale dG[m], and writes f32 dst (2.9% of prefill,
// a full extra M*N HBM round-trip). This EVT folds that into the GEMM epilogue:
//     D_f32[m,n] = acc[m,n] * dG[m]
// ColBroadcast runs in PtrArray mode (ElementInput_ = float*) so the single-group
// dG pointer is supplied via a 1-element device pointer array. Sm90 visitor nodes are
// the SM100/SM120 epilogue's native EVT vocabulary (reused by the sm120 builder).
namespace fusion_ns = cutlass::epilogue::fusion;
using DenseScaleBcast = fusion_ns::Sm90ColBroadcast<
    0, ThreadBlockShape, float* /*per-group ptr array -> PtrArray mode*/, float,
    cute::Stride<cute::_1, cute::_0, cute::_0>>;
using DenseMul = fusion_ns::Sm90Compute<
    cutlass::multiplies, float /*ElementOutput = ElementD*/, float /*ElementCompute*/,
    cutlass::FloatRoundStyle::round_to_nearest>;
using DenseFusion = fusion_ns::Sm90EVT<DenseMul, DenseScaleBcast, fusion_ns::Sm90AccFetch>;

using CollectiveEpilogueDense = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    float, LayoutCTag *, 4 /*AlignmentC (f32)*/,
    float, LayoutCTag *, 4 /*AlignmentD (f32)*/,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    DenseFusion>::CollectiveOp;

using CollectiveMainloopDense = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag *, AlignmentA,
    ElementB, LayoutBTag *, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogueDense::SharedStorage))>,
    MoEKernelSchedule>::CollectiveOp;

using GemmKernelDense = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopDense, CollectiveEpilogueDense>;
using GemmDense = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelDense>;

#define QSUB QSUB_VAL

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
    // MUST check: at large ubatch/context the pool can leave too little VRAM and
    // cudaMalloc fails -> unchecked nullptr write -> sticky illegal memory access
    // (the ub>=2048 crash). On OOM cache a null entry so callers fall back to MMQ.
    if (cudaMalloc(&wc.dB, Bsz) != cudaSuccess) {
        cudaGetLastError(); // clear
        wc.dB = nullptr; wc.dSFB = nullptr;
        fprintf(stderr, "%s: cudaMalloc OOM repacking weights (%zu MiB) - falling back to MMQ for this tensor\n",
                      __func__, (Bsz + SFBsz) >> 20);
        return g_wc.emplace(w_blocks, wc).first->second;
    }
    if (cudaMalloc(&wc.dSFB, SFBsz) != cudaSuccess) {
        cudaGetLastError(); // clear
        cudaFree(wc.dB);
        wc.dB = nullptr; wc.dSFB = nullptr;
        fprintf(stderr, "%s: cudaMalloc OOM repacking weights (%zu MiB) - falling back to MMQ for this tensor\n",
                      __func__, (Bsz + SFBsz) >> 20);
        return g_wc.emplace(w_blocks, wc).first->second;
    }
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
    if (N % 128 != 0) return 0;                 // SFB swizzle atom = 128 rows
    if (K % FP4_SUPPORTED_KMULT != 0) return 0; // FP4 block/SF-atom K granularity
    return 1;
}

extern "C" int ggml_cuda_cutlass_moe_nvfp4(
        const void * w_blocks, const float * src1_sorted, float * dst_sorted,
        const int32_t * tokens_per_expert, int E, int N, int K,
        size_t nb01, size_t nb02, cudaStream_t stream) {
    if (!ggml_cuda_cutlass_moe_nvfp4_supported(E, N, K)) return 1;
#ifdef FC_FP4_MX
    return 1;   // MXFP4: only the fused _ffn path is implemented. The sorted-input
                // path below uses the 2-level act_row_scale/quant_acts glue which has
                // no MX variant, so fall back to MMQ here (never silently wrong).
#endif

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
    if (!wc.dB) return 1; // repack OOM -> caller falls back to MMQ

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
static __global__ void k_build_group_args(
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


// ---------------------------------------------------------------------------
// shared routing cache: gate/up/down of one MoE layer call mul_mat_id with the
// SAME ids tensor -> compute mm_ids_helper once, reuse <=3 times (then forced
// recompute; the allocator may hand the next layer's ids the same address, so
// the use cap is the only thing preventing stale-routing poisoning — 3 matches
// gate/up/down exactly).
//
// These two functions have EXTERNAL linkage and are shared across TUs
// (cutlass-moe-f16.cu calls routing_get). The MXFP4 TU is a #include of this
// file, so their definitions would be emitted twice -> multiple-definition link
// error. Guard the definitions to the NVFP4 TU only; the MXFP4 TU (and the f16
// TU) link against that single copy via the forward declarations below.
// ---------------------------------------------------------------------------
extern "C" void ggml_cuda_moe_routing_get(
        const ggml_cuda_moe_ids_args * a, cudaStream_t stream,
        const int32_t ** ids_src1, const int32_t ** ids_dst,
        const int32_t ** bounds, uint64_t * generation);
extern "C" void ggml_cuda_moe_routing_expire(void);

#ifndef FC_FP4_MX
struct MoeRouting {
    int32_t * buf = nullptr; size_t cap = 0;      // holds ids_src1 + ids_dst + bounds
    int32_t * ids_src1 = nullptr, * ids_dst = nullptr, * bounds = nullptr;
    const int32_t * key_ids = nullptr;
    int key_tok = -1, key_neu = -1, key_E = -1;
    int key_ny = -1, key_sis1 = -1;   // ne11/sis1 CHANGE the ids_src1 gather map:
                                      // gate/up use ne11=1 (broadcast), down uses
                                      // ne11=n_expert_used (per-use rows). MUST key.
    int uses = 0;
    uint64_t generation = 0;
};
static std::mutex g_rt_mtx;
static MoeRouting g_rt;

// Returns device routing arrays for the given ids args (cached or recomputed).
extern "C" void ggml_cuda_moe_routing_get(
        const ggml_cuda_moe_ids_args * a, cudaStream_t stream,
        const int32_t ** ids_src1, const int32_t ** ids_dst,
        const int32_t ** bounds, uint64_t * generation) {
    std::lock_guard<std::mutex> lk(g_rt_mtx);
    const int Mtot = a->n_tokens * a->n_expert_used;
    const bool hit = g_rt.key_ids == a->ids_data && g_rt.key_tok == a->n_tokens &&
                     g_rt.key_neu == a->n_expert_used && g_rt.key_E == a->n_experts &&
                     g_rt.key_ny == a->nchannels_y && g_rt.key_sis1 == a->sis1 &&
                     g_rt.uses > 0 && g_rt.uses < 2;   // only gate->up may share (down
                                                       // differs in ny/sis1); cap=2 so a
                                                       // reused ids ADDRESS from the next
                                                       // layer can never serve stale data
    if (!hit) {
        const size_t need = ((size_t) Mtot * 2 + 3 * a->n_experts + 1) * sizeof(int32_t);
        if (need > g_rt.cap) {
            if (g_rt.buf) cudaFree(g_rt.buf);
            if (cudaMalloc((void **) &g_rt.buf, need + need / 4) != cudaSuccess) {
                cudaGetLastError();
                g_rt.buf = nullptr; g_rt.cap = 0;
                g_rt.key_ids = nullptr; g_rt.uses = 0;
                *ids_src1 = nullptr; *ids_dst = nullptr; *bounds = nullptr;
                return; // OOM -> caller must null-check and fall back to MMQ
            }
            g_rt.cap = need + need / 4;
        }
        g_rt.ids_src1 = g_rt.buf;
        g_rt.ids_dst  = g_rt.buf + Mtot;
        g_rt.bounds   = g_rt.buf + 2 * (size_t) Mtot;
        int32_t * scratch = g_rt.bounds + a->n_experts + 1;   // 2*E ints (counts+cursor)
        // parallel histogram->scan->scatter routing build (Mtot threads) replaces
        // mm_ids_helper (1 warp/expert serial token scan; was 5.8% of prefill GPU
        // time). Within-expert row order is arbitrary — GEMM rows are independent
        // and scatter/gather address through the same maps. E<=1024 (scan block).
        if (a->n_experts <= 1024) {
            ggml_cuda_moe_build_routing(a->ids_data, g_rt.ids_src1, g_rt.ids_dst, g_rt.bounds,
                scratch, a->n_experts, a->n_tokens, a->n_expert_used, a->nchannels_y,
                a->si1, a->sis1, stream);
        } else {
            ggml_cuda_launch_mm_ids_helper(a->ids_data, g_rt.ids_src1, g_rt.ids_dst, g_rt.bounds,
                a->n_experts, a->n_tokens, a->n_expert_used, a->nchannels_y, a->si1, a->sis1, stream);
        }
        g_rt.key_ids = a->ids_data; g_rt.key_tok = a->n_tokens;
        g_rt.key_neu = a->n_expert_used; g_rt.key_E = a->n_experts;
        g_rt.key_ny = a->nchannels_y; g_rt.key_sis1 = a->sis1;
        g_rt.uses = 0;
        g_rt.generation++;
    }
    g_rt.uses++;
    *ids_src1 = g_rt.ids_src1; *ids_dst = g_rt.ids_dst;
    *bounds = g_rt.bounds; *generation = g_rt.generation;
}

// Force the next routing_get to recompute. The whole-FFN path consumes routing
// exactly ONCE per layer, so without this the next layer's ids tensor (often
// the same device address, graph-allocator reuse) would falsely hit the cache
// at uses=1 and serve the previous layer's routing.
extern "C" void ggml_cuda_moe_routing_expire(void) {
    std::lock_guard<std::mutex> lk(g_rt_mtx);
    g_rt.uses = 1000;
}
#endif // FC_FP4_MX (routing cache defined once in the NVFP4 TU)

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
    int32_t  * dInv = nullptr; size_t inv_cap = 0;   // Mtot (inverse permutation)
    void     * ws   = nullptr; size_t ws_cap  = 0;   // GEMM workspace
    // whole-FFN fusion extras (sorted gate/up outputs + second act row scale)
    ElementD * dD1  = nullptr; size_t d1_cap  = 0;   // Mtot*N_gu (gate, sorted)
    ElementD * dD2  = nullptr; size_t d2_cap  = 0;   // Mtot*N_gu (up, sorted)
    float    * dG2  = nullptr; size_t g2_cap  = 0;   // Mtot (post-SwiGLU row scale)
};
static std::mutex g_pf_mtx;
static PrefillScratch g_pf;

static void * grow(void ** p, size_t * cap, size_t need) {
    if (need > *cap) {
        if (*p) cudaFree(*p);
        size_t sz = need + need / 4;            // 25% headroom to damp regrowth
        if (cudaMalloc(p, sz) != cudaSuccess) { // OOM: clear sticky error, signal caller
            cudaGetLastError();
            *p = nullptr; *cap = 0;
            return nullptr;
        }
        *cap = sz;
    }
    return *p;
}

// A-operand reuse cache (file scope so the dense path can invalidate it):
// gate and up projections of one layer consume the SAME activations with the
// SAME routing -> identical quantized A/SFA/G. Skip the quant pass on reuse.
static const float * s_a_src1 = nullptr;
static uint64_t s_a_gen = 0; static int s_a_K = -1, s_a_M = -1;

// Dense-path A-reuse cache: Q/K/V attention projections consume the SAME
// src1 activations (same M,K) back-to-back -> identical quantized A/SFA/G.
// Invalidated whenever the MoE paths clobber the shared dAq/dSFA/dG scratch.
static const float * s_d_src1 = nullptr;
static int s_d_K = -1, s_d_M = -1;

// forward decl (defined below)
extern "C" int ggml_cuda_cutlass_moe_nvfp4_prefill(
        const void * w_blocks, const float * src1, const ggml_cuda_moe_ids_args * ids,
        float * dst, int E, int N, int K, int Mtot, int64_t s11, int64_t s1,
        size_t nb01, size_t nb02, const float * fuse_w, int accum_neu, cudaStream_t stream);

// ===================== dense NVFP4 path (E=1, identity gather) ==============
// Routes large-M dense NVFP4 matmuls (attention projections when the model
// quantizes them to NVFP4) through the same block-scaled FP4 tensor-core GEMM.
// MMQ handles these by unpacking to int8 tensor cores; FP4 cores are ~2x.
extern "C" int ggml_cuda_cutlass_dense_nvfp4(
        const void * w_blocks, const float * src1, float * dst,
        int N, int K, int M, int64_t s11, int64_t s1,
        size_t nb01, size_t nb02, cudaStream_t stream) {
    if (!ggml_cuda_cutlass_moe_nvfp4_supported(1, N, K)) return 1;
    if (M <= 0) return 1;
    if ((size_t) K * sizeof(float) + 128 > 96 * 1024) return 1; // gather smem cap
    if (s1 != N) return 1;                 // direct rowscaled epilogue needs contiguous dst
    std::lock_guard<std::mutex> lk(g_pf_mtx);
    const WeightCache & wc = get_repacked_weights(w_blocks, 1, N, K, nb01, nb02, stream);
    if (!wc.dB) return 1; // repack OOM -> caller falls back to MMQ
    // scratch shared with the MoE prefill path (same stream => ordered, safe),
    // but the A-reuse cache must be invalidated since dAq is overwritten here.
    s_a_src1 = nullptr;
    if (1 > g_pf.E_cap) return 1;          // E-cap scratch built lazily by MoE path; E>=1 always true after first MoE call
    const size_t sf_rows = (size_t) M + 128;
    grow((void **) &g_pf.dAq,  &g_pf.aq_cap,  (size_t) M * (K / 2));
    grow((void **) &g_pf.dSFA, &g_pf.sfa_cap, sf_rows * (K / QSUB));
    grow((void **) &g_pf.dG,   &g_pf.g_cap,   (size_t) M * sizeof(float));
    grow((void **) &g_pf.dD,   &g_pf.d_cap,   (size_t) M * N * sizeof(ElementD));
    if (!g_pf.dAq || !g_pf.dSFA || !g_pf.dG || !g_pf.dD) return 1; // scratch OOM -> fall back to MMQ
    // device bounds/sf_off for E=1: [0, M], [0]
    static int32_t * dB1 = nullptr;
    if (!dB1 && cudaMalloc(&dB1, 3 * sizeof(int32_t)) != cudaSuccess) {
        cudaGetLastError(); dB1 = nullptr; return 1; // OOM -> fall back
    }
    // Q/K/V projections share src1 -> reuse quantized A operand across calls
    const bool d_reuse = s_d_src1 == src1 && s_d_K == K && s_d_M == M;
    if (!d_reuse) {
        const int32_t h[3] = {0, M, 0};
        cudaMemcpyAsync(dB1, h, sizeof(h), cudaMemcpyHostToDevice, stream);
        cudaMemsetAsync(g_pf.dSFA, 0, sf_rows * (K / QSUB), stream);
        ggml_cuda_nvfp4_gather_quant(src1, /*ids_src1=*/nullptr, dB1, dB1 + 2,
                                     g_pf.dG, g_pf.dAq, g_pf.dSFA, 1, M, K, s11, stream);
        s_d_src1 = src1; s_d_K = K; s_d_M = M;
    }
    k_build_group_args<<<1, 1, 0, stream>>>(
        dB1, dB1 + 2, (const uint8_t *) wc.dB, (const uint8_t *) wc.dSFB,
        g_pf.dAq, g_pf.dSFA, g_pf.dD,
        g_pf.ps, g_pf.pA, g_pf.pB, g_pf.pC, g_pf.pD, g_pf.pSFA, g_pf.pSFB,
        g_pf.sa, g_pf.sb, g_pf.sc, g_pf.sd, g_pf.la, g_pf.lb, 1, N, K);
    static cutlass::KernelHardwareInfo hw = [] {
        cutlass::KernelHardwareInfo h; h.device_id = 0;
        h.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
        return h;
    }();
    // ---- FUSED EVT path: D_f32[m,n] = acc[m,n] * dG[m], written straight to dst.
    // Folds the standalone k_bf16_to_f32_rowscaled dequant (2.9% of prefill, a full
    // M*N HBM round-trip) into the GEMM epilogue. Escape hatch reverts to two-step. ----
    static const bool evt_off = getenv("GGML_CUDA_DENSE_EVT_OFF") != nullptr;
    if (!evt_off) {
        // 1-element (E=1) device pointer arrays for the dst output and per-row scale.
        static float **      dPD    = nullptr; // ptr_D array  -> dst
        static const float ** dScale = nullptr; // ColBroadcast ptr_col array -> dG
        if (!dPD    && cudaMalloc(&dPD,    sizeof(float *)) != cudaSuccess) { cudaGetLastError(); dPD = nullptr; }
        if (!dScale && cudaMalloc(&dScale, sizeof(float *)) != cudaSuccess) { cudaGetLastError(); dScale = nullptr; }
        if (dPD && dScale) {
            float *       hPD[1]    = { dst };
            const float * hScale[1] = { g_pf.dG };
            cudaMemcpyAsync(dPD,    hPD,    sizeof(float *), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(dScale, hScale, sizeof(float *), cudaMemcpyHostToDevice, stream);
            typename GemmDense::Arguments args;
            args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
            args.problem_shape = {1, g_pf.ps, nullptr};
            args.mainloop = { g_pf.pA, g_pf.sa, g_pf.pB, g_pf.sb, g_pf.pSFA, g_pf.la, g_pf.pSFB, g_pf.lb };
            // EVT tree Sm90EVT<Mul, ColBroadcast, AccFetch> -> Args {ColBroadcast, AccFetch, Mul}
            args.epilogue.thread = { { dScale, 0.0f, {} }, {}, {} };
            args.epilogue.ptr_C = (const float **) g_pf.pC; // unused (no source load), nullptr entries
            args.epilogue.dC    = (typename GemmKernelDense::CollectiveEpilogue::StrideC) g_pf.sc;
            args.epilogue.ptr_D = dPD;
            args.epilogue.dD    = (typename GemmKernelDense::CollectiveEpilogue::StrideD) g_pf.sd;
            args.hw_info = hw;
            GemmDense gemm;
            const size_t wssz = GemmDense::get_workspace_size(args);
            grow(&g_pf.ws, &g_pf.ws_cap, wssz ? wssz : 1);
            if (gemm.can_implement(args) == cutlass::Status::kSuccess &&
                gemm.initialize(args, g_pf.ws, stream) == cutlass::Status::kSuccess &&
                gemm.run(stream) == cutlass::Status::kSuccess) {
                return 0;
            }
            // EVT can_implement/run failed -> fall through to proven two-step path below.
        }
    }
    typename Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
    args.problem_shape = {1, g_pf.ps, nullptr};
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
    ggml_cuda_bf16_to_f32_rowscaled((const uint16_t *) g_pf.dD, dst, g_pf.dG, M, N, stream);
    return 0;
}

extern "C" int ggml_cuda_cutlass_moe_nvfp4_prefill(
        const void * w_blocks, const float * src1, const ggml_cuda_moe_ids_args * ids,
        float * dst, int E, int N, int K, int Mtot, int64_t s11, int64_t s1,
        size_t nb01, size_t nb02, const float * fuse_w, int accum_neu, cudaStream_t stream) {
    if (!ggml_cuda_cutlass_moe_nvfp4_supported(E, N, K)) return 1;
    if (Mtot <= 0 || E > 1024) return 1;
    if ((size_t) K * sizeof(float) + 128 > 96 * 1024) return 1; // gather kernel smem cap (sm120: 99KB)

    const int32_t * ids_src1; const int32_t * ids_dst; const int32_t * expert_bounds;
    uint64_t routing_gen;
    ggml_cuda_moe_routing_get(ids, stream, &ids_src1, &ids_dst, &expert_bounds, &routing_gen);
    if (!ids_src1) return 1; // routing OOM -> fall back to MMQ

    std::lock_guard<std::mutex> lk(g_pf_mtx);
    const WeightCache & wc = get_repacked_weights(w_blocks, E, N, K, nb01, nb02, stream);
    if (!wc.dB) return 1; // repack OOM -> caller falls back to MMQ

    // ---- scratch (grow-only; steady-state = zero allocations) ----
    if (E > g_pf.E_cap) {
        if (g_pf.args_blob) cudaFree(g_pf.args_blob);
        const size_t per_e =
            sizeof(typename ProblemShape::UnderlyingProblemShape) +
            2 * sizeof(void *) + 2 * sizeof(void *) + 2 * sizeof(void *) +
            sizeof(StrideA) + sizeof(StrideB) + sizeof(StrideC) + sizeof(StrideD) +
            sizeof(LayoutSFA) + sizeof(LayoutSFB) + sizeof(int32_t);
        const int Ecap = E;
        if (cudaMalloc(&g_pf.args_blob, (per_e + 64) * (size_t) Ecap) != cudaSuccess) { // slack for 16B alignment carving
            cudaGetLastError(); g_pf.args_blob = nullptr; return 1; // OOM -> fall back
        }
        uint8_t * base = (uint8_t *) g_pf.args_blob;
        auto carve = [&](size_t bytes) { void * r = base; base += (bytes + 15) & ~size_t(15); return r; };
        g_pf.ps   = (typename ProblemShape::UnderlyingProblemShape *) carve(sizeof(*g_pf.ps) * Ecap);
        g_pf.pA   = (const ElementAData **) carve(sizeof(void *) * Ecap);
        g_pf.pB   = (const ElementBData **) carve(sizeof(void *) * Ecap);
        g_pf.pC   = (const ElementC **)     carve(sizeof(void *) * Ecap);
        g_pf.pD   = (ElementD **)           carve(sizeof(void *) * Ecap);
        g_pf.pSFA = (const ElementSF **)    carve(sizeof(void *) * Ecap);
        g_pf.pSFB = (const ElementSF **)    carve(sizeof(void *) * Ecap);
        g_pf.sa   = (StrideA *)   carve(sizeof(StrideA) * Ecap);
        g_pf.sb   = (StrideB *)   carve(sizeof(StrideB) * Ecap);
        g_pf.sc   = (StrideC *)   carve(sizeof(StrideC) * Ecap);
        g_pf.sd   = (StrideD *)   carve(sizeof(StrideD) * Ecap);
        g_pf.la   = (LayoutSFA *) carve(sizeof(LayoutSFA) * Ecap);
        g_pf.lb   = (LayoutSFB *) carve(sizeof(LayoutSFB) * Ecap);
        g_pf.sf_off = (int32_t *) carve(sizeof(int32_t) * Ecap);
        g_pf.E_cap = E;
    }
    const size_t sf_rows = (size_t) Mtot + 128 * (size_t) E;    // worst-case round-up padding
    grow((void **) &g_pf.dAq,  &g_pf.aq_cap,  (size_t) Mtot * (K / 2));
    grow((void **) &g_pf.dSFA, &g_pf.sfa_cap, sf_rows * (K / QSUB));
    grow((void **) &g_pf.dG,   &g_pf.g_cap,   (size_t) Mtot * sizeof(float));
    grow((void **) &g_pf.dD,   &g_pf.d_cap,   (size_t) Mtot * N * sizeof(ElementD));
    if (!g_pf.dAq || !g_pf.dSFA || !g_pf.dG || !g_pf.dD) return 1; // scratch OOM -> fall back to MMQ

    // ---- device-side prep: sf_offsets -> gather+quant -> group args (no syncs) ----
    // A-operand reuse: gate and up projections of one layer consume the SAME
    // activations with the SAME routing -> identical quantized A/SFA/G. Skip
    // the whole quant pass on the second call (measured: k_gather_quant was
    // ~5-8% of prefill GPU time; this halves it). Cache vars at file scope
    // (invalidated by the dense path which shares the scratch buffers).
    const bool a_reusable = s_a_src1 == src1 && s_a_gen == routing_gen &&
                            s_a_K == K && s_a_M == Mtot;
    if (!a_reusable) {
        ggml_cuda_nvfp4_sf_offsets(expert_bounds, g_pf.sf_off, E, stream);
        cudaMemsetAsync(g_pf.dSFA, 0, sf_rows * (K / QSUB), stream);
        ggml_cuda_nvfp4_gather_quant(src1, ids_src1, expert_bounds, g_pf.sf_off,
                                     g_pf.dG, g_pf.dAq, g_pf.dSFA, E, Mtot, K, s11, stream);
        s_a_src1 = src1; s_a_gen = routing_gen; s_a_K = K; s_a_M = Mtot;
        s_d_src1 = nullptr; // dense A-cache clobbered
    }
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

    // ---- fused expert-sum + rescale + bf16->f32 straight into unsorted dst ----
    if (accum_neu > 0 && accum_neu <= 16) {
        // no-atomic path: invert permutation once, then per-token register-accum
        // gather (each dst element written exactly once; no memset, no atomics)
        grow((void **) &g_pf.dInv, &g_pf.inv_cap, (size_t) Mtot * sizeof(int32_t));
        if (!g_pf.dInv) return 1; // scratch OOM -> fall back to MMQ
        ggml_cuda_moe_invert_ids(ids_dst, g_pf.dInv, Mtot, stream);
        ggml_cuda_nvfp4_gather_accum_rowscaled(g_pf.dD, g_pf.dG, g_pf.dInv, dst,
                                               Mtot / accum_neu, N, s1, fuse_w, accum_neu,
                                               nullptr, 0, stream);
    } else {
        if (accum_neu > 0) {
            cudaMemsetAsync(dst, 0, (size_t) (Mtot / accum_neu) * s1 * sizeof(float), stream);
        }
        ggml_cuda_nvfp4_scatter_rowscaled(g_pf.dD, g_pf.dG, ids_dst, dst, Mtot, N, s1, fuse_w, accum_neu, stream);
    }
    return 0;
}

// ===================== WHOLE-FFN fused path (2026-07-02) ====================
// gate GEMM -> fused SwiGLU+NVFP4 requant ON SORTED ROWS -> down GEMM.
// Eliminates per-layer: 2x scatter (gate/up unsort), silu kernel, down-proj
// re-gather+quant, and all intermediate f32 materialization. Routing is built
// ONCE (gate/up ids); the down GEMM consumes sorted rows directly and the
// final gather-accum epilogue lands in the unsorted output with routing
// weights applied. ids_dst is ne11-independent (flat it*neu+iex), so the
// gate-routing ids_dst is exactly the down-scatter target and fuse_w index.
static int run_group_gemm_locked(const WeightCache & wc, const int32_t * bounds,
        const int32_t * sf_off, const uint8_t * dAq, const uint8_t * dSFA,
        ElementD * dOut, int E, int N, int K, cudaStream_t stream) {
    {
        const int threads = 128;
        const int blocks = (E + threads - 1) / threads;
        k_build_group_args<<<blocks, threads, 0, stream>>>(
            bounds, sf_off, (const uint8_t *) wc.dB, (const uint8_t *) wc.dSFB,
            dAq, dSFA, dOut,
            g_pf.ps, g_pf.pA, g_pf.pB, g_pf.pC, g_pf.pD, g_pf.pSFA, g_pf.pSFB,
            g_pf.sa, g_pf.sb, g_pf.sc, g_pf.sd, g_pf.la, g_pf.lb, E, N, K);
    }
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
    return 0;
}

extern "C" int ggml_cuda_cutlass_moe_nvfp4_ffn(
        const void * w_gate, const void * w_up, const void * w_down,
        const float * src1, const ggml_cuda_moe_ids_args * ids,
        float * dst, int E, int N_gu, int K, int N_out, int Mtot,
        int64_t s11, int64_t s1,
    size_t g_nb01, size_t g_nb02, size_t u_nb01, size_t u_nb02,
    size_t d_nb01, size_t d_nb02,
    const float * fuse_w, int accum_neu,
    const float * residual, int64_t res_s1, cudaStream_t stream) {
    if (!ggml_cuda_cutlass_moe_nvfp4_supported(E, N_gu, K) ||
        !ggml_cuda_cutlass_moe_nvfp4_supported(E, N_out, N_gu)) return 1;
    if (Mtot <= 0 || E > 1024) return 1;
    const size_t max_kf = (size_t) (K > N_gu ? K : N_gu);
    if (max_kf * sizeof(float) + 128 > 96 * 1024) return 1; // gather/swiglu smem cap

    const int32_t * ids_src1; const int32_t * ids_dst; const int32_t * bounds;
    uint64_t routing_gen; (void) routing_gen;
    ggml_cuda_moe_routing_get(ids, stream, &ids_src1, &ids_dst, &bounds, &routing_gen);
    if (!ids_src1) return 1; // routing OOM -> fall back to MMQ
    // this path consumes routing exactly once per layer -> kill the cache entry
    // so the next layer's ids (possibly same address) can never false-hit
    ggml_cuda_moe_routing_expire();

    std::lock_guard<std::mutex> lk(g_pf_mtx);
    const WeightCache & wcg = get_repacked_weights(w_gate, E, N_gu,  K,    g_nb01, g_nb02, stream);
    const WeightCache & wcu = get_repacked_weights(w_up,   E, N_gu,  K,    u_nb01, u_nb02, stream);
    const WeightCache & wcd = get_repacked_weights(w_down, E, N_out, N_gu, d_nb01, d_nb02, stream);
    if (!wcg.dB || !wcu.dB || !wcd.dB) return 1; // repack OOM -> caller falls back

    if (E > g_pf.E_cap) {
        // args blob built by the generic prefill path; build it here the same way
        if (g_pf.args_blob) cudaFree(g_pf.args_blob);
        const size_t per_e =
            sizeof(typename ProblemShape::UnderlyingProblemShape) +
            6 * sizeof(void *) +
            sizeof(StrideA) + sizeof(StrideB) + sizeof(StrideC) + sizeof(StrideD) +
            sizeof(LayoutSFA) + sizeof(LayoutSFB) + sizeof(int32_t);
        const int Ecap = E;
        if (cudaMalloc(&g_pf.args_blob, (per_e + 64) * (size_t) Ecap) != cudaSuccess) {
            cudaGetLastError(); g_pf.args_blob = nullptr; return 1; // OOM -> fall back
        }
        uint8_t * base = (uint8_t *) g_pf.args_blob;
        auto carve = [&](size_t bytes) { void * r = base; base += (bytes + 15) & ~size_t(15); return r; };
        g_pf.ps   = (typename ProblemShape::UnderlyingProblemShape *) carve(sizeof(*g_pf.ps) * Ecap);
        g_pf.pA   = (const ElementAData **) carve(sizeof(void *) * Ecap);
        g_pf.pB   = (const ElementBData **) carve(sizeof(void *) * Ecap);
        g_pf.pC   = (const ElementC **)     carve(sizeof(void *) * Ecap);
        g_pf.pD   = (ElementD **)           carve(sizeof(void *) * Ecap);
        g_pf.pSFA = (const ElementSF **)    carve(sizeof(void *) * Ecap);
        g_pf.pSFB = (const ElementSF **)    carve(sizeof(void *) * Ecap);
        g_pf.sa   = (StrideA *)   carve(sizeof(StrideA) * Ecap);
        g_pf.sb   = (StrideB *)   carve(sizeof(StrideB) * Ecap);
        g_pf.sc   = (StrideC *)   carve(sizeof(StrideC) * Ecap);
        g_pf.sd   = (StrideD *)   carve(sizeof(StrideD) * Ecap);
        g_pf.la   = (LayoutSFA *) carve(sizeof(LayoutSFA) * Ecap);
        g_pf.lb   = (LayoutSFB *) carve(sizeof(LayoutSFB) * Ecap);
        g_pf.sf_off = (int32_t *) carve(sizeof(int32_t) * Ecap);
        g_pf.E_cap = E;
    }
    const size_t sf_rows = (size_t) Mtot + 128 * (size_t) E;
    grow((void **) &g_pf.dAq,  &g_pf.aq_cap,  (size_t) Mtot * (max_kf / 2));
    grow((void **) &g_pf.dSFA, &g_pf.sfa_cap, sf_rows * (max_kf / QSUB));
    grow((void **) &g_pf.dG,   &g_pf.g_cap,   (size_t) Mtot * sizeof(float));
    grow((void **) &g_pf.dG2,  &g_pf.g2_cap,  (size_t) Mtot * sizeof(float));
    grow((void **) &g_pf.dD1,  &g_pf.d1_cap,  (size_t) Mtot * N_gu * sizeof(ElementD));
    grow((void **) &g_pf.dD2,  &g_pf.d2_cap,  (size_t) Mtot * N_gu * sizeof(ElementD));
    grow((void **) &g_pf.dD,   &g_pf.d_cap,   (size_t) Mtot * N_out * sizeof(ElementD));
    if (!g_pf.dAq || !g_pf.dSFA || !g_pf.dG || !g_pf.dG2 || !g_pf.dD1 || !g_pf.dD2 || !g_pf.dD) return 1; // scratch OOM -> fall back to MMQ
    s_a_src1 = nullptr;   // we clobber dAq/dSFA/dG below
    s_d_src1 = nullptr;   // dense A-cache clobbered too

    // ---- input quant (once for gate AND up) ----
    ggml_cuda_nvfp4_sf_offsets(bounds, g_pf.sf_off, E, stream);
    cudaMemsetAsync(g_pf.dSFA, 0, sf_rows * (K / QSUB), stream);
    ggml_cuda_nvfp4_gather_quant(src1, ids_src1, bounds, g_pf.sf_off,
                                 g_pf.dG, g_pf.dAq, g_pf.dSFA, E, Mtot, K, s11, stream);
    // ---- gate & up GEMMs (sorted bf16 outputs) ----
    // NOTE: merging these into one 2E-group GEMM (run_group_gemm_gu_locked) was
    // measured -3.9% (28.1k vs 29.2k) — the grouped-GEMM group-iteration overhead
    // outweighs the halved launch count. Two separate E-group GEMMs pipeline
    // better on sm_120. Kept as separate calls.
    int rc;
    if ((rc = run_group_gemm_locked(wcg, bounds, g_pf.sf_off, g_pf.dAq, g_pf.dSFA, g_pf.dD1, E, N_gu, K, stream)) != 0) return rc;
    if ((rc = run_group_gemm_locked(wcu, bounds, g_pf.sf_off, g_pf.dAq, g_pf.dSFA, g_pf.dD2, E, N_gu, K, stream)) != 0) return rc;
    // ---- fused SwiGLU + requant on sorted rows (clobbers dAq/dSFA, stream-ordered) ----
    cudaMemsetAsync(g_pf.dSFA, 0, sf_rows * (N_gu / QSUB), stream);
    ggml_cuda_nvfp4_swiglu_quant(g_pf.dD1, g_pf.dD2, g_pf.dG, bounds, g_pf.sf_off,
                                 g_pf.dG2, g_pf.dAq, g_pf.dSFA, E, Mtot, N_gu, stream);
    // ---- down GEMM ----
    if ((rc = run_group_gemm_locked(wcd, bounds, g_pf.sf_off, g_pf.dAq, g_pf.dSFA, g_pf.dD, E, N_out, N_gu, stream)) != 0) return rc;
    // ---- epilogue: rescale + routing weights + expert sum into unsorted dst ----
    if (accum_neu > 0 && accum_neu <= 16) {
        grow((void **) &g_pf.dInv, &g_pf.inv_cap, (size_t) Mtot * sizeof(int32_t));
        if (!g_pf.dInv) return 1; // scratch OOM -> fall back to MMQ
        ggml_cuda_moe_invert_ids(ids_dst, g_pf.dInv, Mtot, stream);
        ggml_cuda_nvfp4_gather_accum_rowscaled(g_pf.dD, g_pf.dG2, g_pf.dInv, dst,
                                               Mtot / accum_neu, N_out, s1, fuse_w, accum_neu,
                                               residual, res_s1, stream);
    } else {
        if (accum_neu > 0) {
            cudaMemsetAsync(dst, 0, (size_t) (Mtot / accum_neu) * s1 * sizeof(float), stream);
        }
        ggml_cuda_nvfp4_scatter_rowscaled(g_pf.dD, g_pf.dG2, ids_dst, dst, Mtot, N_out, s1, fuse_w, accum_neu, stream);
    }
    return 0;
}
