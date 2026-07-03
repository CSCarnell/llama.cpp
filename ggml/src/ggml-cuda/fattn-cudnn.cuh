#pragma once

#include "common.cuh"

// FCFA-cuDNN: cuDNN SDPA (Blackwell FA3-class warp-specialized/TMA kernels) for
// large-batch prefill flash-attention. Only fires when the mask is proven to be
// exactly bottom-right causal (GPU-side analysis, cached per graph eval);
// everything else falls back to the built-in kernels.
//
// Returns true if the op was handled by cuDNN, false -> caller dispatches as usual.
#ifdef GGML_CUDA_USE_CUDNN
bool ggml_cuda_flash_attn_ext_cudnn_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// Call at the start of every backend graph_compute: invalidates the cached
// per-eval mask-band verdict (mask contents change between ubatches).
void ggml_cuda_fattn_cudnn_begin_graph(void);
#else
static inline bool ggml_cuda_flash_attn_ext_cudnn_try(ggml_backend_cuda_context &, ggml_tensor *) { return false; }
static inline void ggml_cuda_fattn_cudnn_begin_graph(void) {}
#endif
