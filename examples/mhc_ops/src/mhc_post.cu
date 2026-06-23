// Baseline CUDA implementation of the MHC post operator.
//
// Assumed formula:
//   new_residual_i = sum_j comb_mix[i, j] * residual_j + post_mix_i * x
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

__global__ void mhc_post_baseline_kernel(
    const mhc_bfloat16* __restrict__ x,
    const mhc_bfloat16* __restrict__ residual,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    int t_tokens,
    int hc_mult,
    int hidden_size,
    mhc_bfloat16* __restrict__ new_residual) {
  const int total = t_tokens * hc_mult * hidden_size;
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }

  const int h = hidden_size;
  const int c = hc_mult;
  const int hidden = linear % h;
  const int channel = (linear / h) % c;
  const int token = linear / (c * h);

  const int residual_base = token * c * h;
  const int comb_base = token * c * c + channel * c;
  float acc =
      post_mix[token * c + channel] * mhc_bf16_to_float(x[token * h + hidden]);

  for (int src_channel = 0; src_channel < c; ++src_channel) {
    const float weight = comb_mix[comb_base + src_channel];
    const float value = mhc_bf16_to_float(
        residual[residual_base + src_channel * h + hidden]);
    acc += weight * value;
  }

  new_residual[linear] = mhc_float_to_bf16(acc);
}

} // namespace

extern "C" mhc_error_t mhc_post_cuda_launch(
    const mhc_bfloat16* x,
    const mhc_bfloat16* residual,
    const float* post_mix,
    const float* comb_mix,
    int t_tokens,
    int hc_mult,
    int hidden_size,
    mhc_bfloat16* new_residual,
    mhc_stream_t stream) {
  const int threads = 256;
  const int total = t_tokens * hc_mult * hidden_size;
  const int blocks = (total + threads - 1) / threads;

  mhc_post_baseline_kernel<<<blocks, threads, 0, stream>>>(
      x,
      residual,
      post_mix,
      comb_mix,
      t_tokens,
      hc_mult,
      hidden_size,
      new_residual);
  return MHC_GET_LAST_ERROR();
}
