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


__global__ void mhc_post_c4_h4096_kernel(
    const mhc_bfloat16* __restrict__ x,
    const mhc_bfloat16* __restrict__ residual,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    int total,
    mhc_bfloat16* __restrict__ new_residual) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }

  const int hidden = linear & 4095;
  const int token = linear >> 12;
  const int base = (token << 14) + hidden;
  const int mix_base = token << 2;
  const int comb_base = token << 4;

  const float xval = mhc_bf16_to_float(x[linear]);
  const float r0 = mhc_bf16_to_float(residual[base]);
  const float r1 = mhc_bf16_to_float(residual[base + 4096]);
  const float r2 = mhc_bf16_to_float(residual[base + 8192]);
  const float r3 = mhc_bf16_to_float(residual[base + 12288]);

  const float p0 = post_mix[mix_base];
  const float p1 = post_mix[mix_base + 1];
  const float p2 = post_mix[mix_base + 2];
  const float p3 = post_mix[mix_base + 3];

  float a0 = p0 * xval;
  a0 = fmaf(comb_mix[comb_base], r0, a0);
  a0 = fmaf(comb_mix[comb_base + 1], r1, a0);
  a0 = fmaf(comb_mix[comb_base + 2], r2, a0);
  a0 = fmaf(comb_mix[comb_base + 3], r3, a0);

  float a1 = p1 * xval;
  a1 = fmaf(comb_mix[comb_base + 4], r0, a1);
  a1 = fmaf(comb_mix[comb_base + 5], r1, a1);
  a1 = fmaf(comb_mix[comb_base + 6], r2, a1);
  a1 = fmaf(comb_mix[comb_base + 7], r3, a1);

  float a2 = p2 * xval;
  a2 = fmaf(comb_mix[comb_base + 8], r0, a2);
  a2 = fmaf(comb_mix[comb_base + 9], r1, a2);
  a2 = fmaf(comb_mix[comb_base + 10], r2, a2);
  a2 = fmaf(comb_mix[comb_base + 11], r3, a2);

  float a3 = p3 * xval;
  a3 = fmaf(comb_mix[comb_base + 12], r0, a3);
  a3 = fmaf(comb_mix[comb_base + 13], r1, a3);
  a3 = fmaf(comb_mix[comb_base + 14], r2, a3);
  a3 = fmaf(comb_mix[comb_base + 15], r3, a3);

  new_residual[base] = mhc_float_to_bf16(a0);
  new_residual[base + 4096] = mhc_float_to_bf16(a1);
  new_residual[base + 8192] = mhc_float_to_bf16(a2);
  new_residual[base + 12288] = mhc_float_to_bf16(a3);
}

__global__ void mhc_post_c4_generic_kernel(
    const mhc_bfloat16* __restrict__ x,
    const mhc_bfloat16* __restrict__ residual,
    const float* __restrict__ post_mix,
    const float* __restrict__ comb_mix,
    int t_tokens,
    int hidden_size,
    mhc_bfloat16* __restrict__ new_residual) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = t_tokens * hidden_size;
  if (linear >= total) {
    return;
  }

  const int hidden = linear % hidden_size;
  const int token = linear / hidden_size;
  const int base = token * 4 * hidden_size + hidden;
  const int mix_base = token * 4;
  const int comb_base = token * 16;

  const float xval = mhc_bf16_to_float(x[token * hidden_size + hidden]);
  const float r0 = mhc_bf16_to_float(residual[base]);
  const float r1 = mhc_bf16_to_float(residual[base + hidden_size]);
  const float r2 = mhc_bf16_to_float(residual[base + 2 * hidden_size]);
  const float r3 = mhc_bf16_to_float(residual[base + 3 * hidden_size]);

  float a0 = post_mix[mix_base] * xval;
  a0 = fmaf(comb_mix[comb_base], r0, a0);
  a0 = fmaf(comb_mix[comb_base + 1], r1, a0);
  a0 = fmaf(comb_mix[comb_base + 2], r2, a0);
  a0 = fmaf(comb_mix[comb_base + 3], r3, a0);

  float a1 = post_mix[mix_base + 1] * xval;
  a1 = fmaf(comb_mix[comb_base + 4], r0, a1);
  a1 = fmaf(comb_mix[comb_base + 5], r1, a1);
  a1 = fmaf(comb_mix[comb_base + 6], r2, a1);
  a1 = fmaf(comb_mix[comb_base + 7], r3, a1);

  float a2 = post_mix[mix_base + 2] * xval;
  a2 = fmaf(comb_mix[comb_base + 8], r0, a2);
  a2 = fmaf(comb_mix[comb_base + 9], r1, a2);
  a2 = fmaf(comb_mix[comb_base + 10], r2, a2);
  a2 = fmaf(comb_mix[comb_base + 11], r3, a2);

  float a3 = post_mix[mix_base + 3] * xval;
  a3 = fmaf(comb_mix[comb_base + 12], r0, a3);
  a3 = fmaf(comb_mix[comb_base + 13], r1, a3);
  a3 = fmaf(comb_mix[comb_base + 14], r2, a3);
  a3 = fmaf(comb_mix[comb_base + 15], r3, a3);

  new_residual[base] = mhc_float_to_bf16(a0);
  new_residual[base + hidden_size] = mhc_float_to_bf16(a1);
  new_residual[base + 2 * hidden_size] = mhc_float_to_bf16(a2);
  new_residual[base + 3 * hidden_size] = mhc_float_to_bf16(a3);
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
  if (hc_mult == 4) {
    const int total = t_tokens * hidden_size;
    const int blocks = (total + threads - 1) / threads;
    if (hidden_size == 4096) {
      mhc_post_c4_h4096_kernel<<<blocks, threads, 0, stream>>>(
          x, residual, post_mix, comb_mix, total, new_residual);
    } else {
      mhc_post_c4_generic_kernel<<<blocks, threads, 0, stream>>>(
          x, residual, post_mix, comb_mix, t_tokens, hidden_size, new_residual);
    }
  } else {
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
  }
  return MHC_GET_LAST_ERROR();
}
