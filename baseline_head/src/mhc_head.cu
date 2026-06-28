// Baseline CUDA implementation of the MHC head fold operator.
//
// Formula:
//   hidden = sum_c head_mix[c] * residual[c]
//
// This file is intentionally simple and semantic-first. It can be compiled by
// hipcc with "-x hip" on Hygon/ROCm systems.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

using mhc_bfloat16 = hip_bfloat16;
using mhc_error_t = hipError_t;
using mhc_stream_t = hipStream_t;
#define MHC_GET_LAST_ERROR hipGetLastError
#else
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using mhc_bfloat16 = __nv_bfloat16;
using mhc_error_t = cudaError_t;
using mhc_stream_t = cudaStream_t;
#define MHC_GET_LAST_ERROR cudaGetLastError
#endif

namespace {

__device__ __forceinline__ float mhc_bf16_to_float(mhc_bfloat16 value) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return static_cast<float>(value);
#else
  return __bfloat162float(value);
#endif
}

__device__ __forceinline__ mhc_bfloat16 mhc_float_to_bf16(float value) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return mhc_bfloat16(value);
#else
  return __float2bfloat16(value);
#endif
}

__global__ void hc_head_baseline_kernel(
    const mhc_bfloat16* __restrict__ residual,
    const float* __restrict__ head_mix,
    int t_tokens,
    int hc_mult,
    int hidden_size,
    mhc_bfloat16* __restrict__ hidden) {
  const int total = t_tokens * hidden_size;
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }

  const int token = linear / hidden_size;
  const int hidden_idx = linear - token * hidden_size;
  const int residual_base = token * hc_mult * hidden_size;
  const int head_base = token * hc_mult;

  float acc = 0.0f;
  for (int channel = 0; channel < hc_mult; ++channel) {
    const float weight = head_mix[head_base + channel];
    const float value = mhc_bf16_to_float(
        residual[residual_base + channel * hidden_size + hidden_idx]);
    acc += weight * value;
  }

  hidden[linear] = mhc_float_to_bf16(acc);
}

} // namespace

extern "C" mhc_error_t hc_head_cuda_launch(
    const mhc_bfloat16* residual,
    const float* head_mix,
    int t_tokens,
    int hc_mult,
    int hidden_size,
    mhc_bfloat16* hidden,
    mhc_stream_t stream) {
  const int threads = 256;
  const int total = t_tokens * hidden_size;
  const int blocks = (total + threads - 1) / threads;

  hc_head_baseline_kernel<<<blocks, threads, 0, stream>>>(
      residual,
      head_mix,
      t_tokens,
      hc_mult,
      hidden_size,
      hidden);
  return MHC_GET_LAST_ERROR();
}
