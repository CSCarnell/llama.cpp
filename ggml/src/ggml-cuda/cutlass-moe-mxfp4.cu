// MXFP4 variant of the Blackwell grouped-GEMM MoE kernel.
//
// This TU is a thin wrapper: it defines FC_FP4_MX and includes the full NVFP4
// implementation, which is written to compile for BOTH formats. The macro
// switches the CUTLASS element type (nv_float4_t -> mx_float4_t, i.e. e4m3 SF
// block-16 -> e8m0 SF block-32), the SF K-granularity (QSUB 16 -> 32), the
// _supported K-multiple check (64 -> 128), and preprocessor-renames the extern
// "C" entry points + format-specific glue calls to their mxfp4 spellings:
//     ggml_cuda_cutlass_moe_mxfp4_ffn / _supported / (base)
//     ggml_cuda_mxfp4_{repack_weights,gather_quant,swiglu_quant}
// Format-independent glue (sf_offsets, scatter/gather_accum_rowscaled,
// act_row_scale, quant_acts, moe_routing_*) is shared verbatim.
//
// All file-scope helpers in the included TU are `static` (internal linkage), so
// this object file and cutlass-moe-fp4.o never collide at link time.
#define FC_FP4_MX 1
#include "cutlass-moe-fp4.cu"
