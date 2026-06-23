// Baseline CUDA implementation of the MHC pre operator.
//
// This file is intentionally simple and semantic-first. It is meant to be a
// correct baseline that can be hipified or rewritten by GEAK for Hygon DCU.

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

#include <math.h>

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

__device__ __forceinline__ float sigmoidf_stable(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ float block_sum(float value, float* scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__global__ void mhc_pre_baseline_kernel(
    const mhc_bfloat16* __restrict__ residual,
    const float* __restrict__ fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_mult_value,
    int sinkhorn_repeat,
    int t_tokens,
    int hc_mult,
    int hidden_size,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    mhc_bfloat16* __restrict__ layer_input) {
  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  if (token >= t_tokens) {
    return;
  }

  const int c = hc_mult;
  const int h = hidden_size;
  const int ch = c * h;
  const int c3 = c * 2 + c * c;
  const int residual_base = token * ch;

  extern __shared__ float smem[];
  float* mixes = smem;             // [c3]
  float* pre = mixes + c3;         // [c]
  float* comb = pre + c;           // [c * c]
  float* reduction = comb + c * c; // [blockDim.x]

  float local_sqr = 0.0f;
  for (int idx = tid; idx < ch; idx += blockDim.x) {
    const float v = mhc_bf16_to_float(residual[residual_base + idx]);
    local_sqr += v * v;
  }
  const float sqrsum = block_sum(local_sqr, reduction);
  const float rms_scale = rsqrtf(sqrsum / static_cast<float>(ch) + rms_eps);

  for (int mix_idx = 0; mix_idx < c3; ++mix_idx) {
    float local_dot = 0.0f;
    const int fn_base = mix_idx * ch;
    for (int idx = tid; idx < ch; idx += blockDim.x) {
      const float v = mhc_bf16_to_float(residual[residual_base + idx]);
      local_dot += v * fn[fn_base + idx];
    }
    const float dot = block_sum(local_dot, reduction);
    if (tid == 0) {
      mixes[mix_idx] = dot * rms_scale;
    }
    __syncthreads();
  }

  if (tid < c) {
    const int i = tid;
    pre[i] = sigmoidf_stable(mixes[i] * hc_scale[0] + hc_base[i]) + hc_pre_eps;
    post_mix[token * c + i] =
        sigmoidf_stable(mixes[c + i] * hc_scale[1] + hc_base[c + i]) *
        hc_post_mult_value;
  }

  if (tid < c * c) {
    const int idx = tid;
    comb[idx] = mixes[2 * c + idx] * hc_scale[2] + hc_base[2 * c + idx];
  }
  __syncthreads();

  // Softmax over the last dimension of each [c, c] row, then add eps.
  if (tid < c) {
    const int row = tid;
    float row_max = comb[row * c];
    for (int col = 1; col < c; ++col) {
      row_max = fmaxf(row_max, comb[row * c + col]);
    }

    float row_sum = 0.0f;
    for (int col = 0; col < c; ++col) {
      const float v = expf(comb[row * c + col] - row_max);
      comb[row * c + col] = v;
      row_sum += v;
    }

    for (int col = 0; col < c; ++col) {
      comb[row * c + col] = comb[row * c + col] / row_sum + hc_sinkhorn_eps;
    }
  }
  __syncthreads();

  // First column normalization, matching PyTorch sum(dim=-2).
  if (tid < c) {
    const int col = tid;
    float col_sum = 0.0f;
    for (int row = 0; row < c; ++row) {
      col_sum += comb[row * c + col];
    }
    const float denom = col_sum + hc_sinkhorn_eps;
    for (int row = 0; row < c; ++row) {
      comb[row * c + col] /= denom;
    }
  }
  __syncthreads();

  for (int repeat = 1; repeat < sinkhorn_repeat; ++repeat) {
    if (tid < c) {
      const int row = tid;
      float row_sum = 0.0f;
      for (int col = 0; col < c; ++col) {
        row_sum += comb[row * c + col];
      }
      const float denom = row_sum + hc_sinkhorn_eps;
      for (int col = 0; col < c; ++col) {
        comb[row * c + col] /= denom;
      }
    }
    __syncthreads();

    if (tid < c) {
      const int col = tid;
      float col_sum = 0.0f;
      for (int row = 0; row < c; ++row) {
        col_sum += comb[row * c + col];
      }
      const float denom = col_sum + hc_sinkhorn_eps;
      for (int row = 0; row < c; ++row) {
        comb[row * c + col] /= denom;
      }
    }
    __syncthreads();
  }

  for (int idx = tid; idx < c * c; idx += blockDim.x) {
    comb_mix[token * c * c + idx] = comb[idx];
  }

  for (int hidden = tid; hidden < h; hidden += blockDim.x) {
    float acc = 0.0f;
    for (int channel = 0; channel < c; ++channel) {
      const float r =
          mhc_bf16_to_float(residual[residual_base + channel * h + hidden]);
      acc += pre[channel] * r;
    }
    layer_input[token * h + hidden] = mhc_float_to_bf16(acc);
  }
}

} // namespace

__global__ void mhc_pre_c4_h4096_kernel(
    const mhc_bfloat16* __restrict__ residual,
    const float* __restrict__ fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_mult_value,
    int t_tokens,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    mhc_bfloat16* __restrict__ layer_input) {
  constexpr int c = 4;
  constexpr int h = 4096;
  constexpr int ch = c * h;
  constexpr int c3 = c * 2 + c * c;
  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  if (token >= t_tokens) return;

  extern __shared__ float smem[];
  float* red = smem;                 // [25 * blockDim.x]
  float* mixes = red + 25 * blockDim.x; // [24]
  float* pre = mixes + c3;           // [4]
  float* comb = pre + c;             // [16]

  const int residual_base = token * ch;
  float sqr = 0.0f;
  float acc[c3];
#pragma unroll
  for (int m = 0; m < c3; ++m) acc[m] = 0.0f;

  for (int idx = tid; idx < ch; idx += blockDim.x) {
    const float v = mhc_bf16_to_float(residual[residual_base + idx]);
    sqr += v * v;
#pragma unroll
    for (int m = 0; m < c3; ++m) {
      acc[m] = fmaf(v, fn[m * ch + idx], acc[m]);
    }
  }

  red[tid] = sqr;
#pragma unroll
  for (int m = 0; m < c3; ++m) {
    red[(m + 1) * blockDim.x + tid] = acc[m];
  }
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
#pragma unroll
      for (int m = 0; m < 25; ++m) {
        red[m * blockDim.x + tid] += red[m * blockDim.x + tid + stride];
      }
    }
    __syncthreads();
  }

  const float rms_scale = rsqrtf(red[0] * (1.0f / static_cast<float>(ch)) + rms_eps);
  if (tid < c3) {
    mixes[tid] = red[(tid + 1) * blockDim.x] * rms_scale;
  }
  __syncthreads();

  if (tid < c) {
    const int i = tid;
    pre[i] = sigmoidf_stable(mixes[i] * hc_scale[0] + hc_base[i]) + hc_pre_eps;
    post_mix[token * c + i] = sigmoidf_stable(mixes[c + i] * hc_scale[1] + hc_base[c + i]) * hc_post_mult_value;
  }
  if (tid < c * c) {
    const int idx = tid;
    comb[idx] = mixes[2 * c + idx] * hc_scale[2] + hc_base[2 * c + idx];
  }
  __syncthreads();

  if (tid < c) {
    const int row = tid;
    float v0 = comb[row * c + 0];
    float v1 = comb[row * c + 1];
    float v2 = comb[row * c + 2];
    float v3 = comb[row * c + 3];
    const float mx = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
    v0 = expf(v0 - mx); v1 = expf(v1 - mx); v2 = expf(v2 - mx); v3 = expf(v3 - mx);
    const float inv = 1.0f / (v0 + v1 + v2 + v3);
    comb[row * c + 0] = v0 * inv + hc_sinkhorn_eps;
    comb[row * c + 1] = v1 * inv + hc_sinkhorn_eps;
    comb[row * c + 2] = v2 * inv + hc_sinkhorn_eps;
    comb[row * c + 3] = v3 * inv + hc_sinkhorn_eps;
  }
  __syncthreads();

  if (tid < c) {
    const int col = tid;
    const float denom = comb[col] + comb[c + col] + comb[2 * c + col] + comb[3 * c + col] + hc_sinkhorn_eps;
    comb[col] /= denom;
    comb[c + col] /= denom;
    comb[2 * c + col] /= denom;
    comb[3 * c + col] /= denom;
  }
  __syncthreads();

  if (tid < c) {
    const int row = tid * c;
    const float denom = comb[row] + comb[row + 1] + comb[row + 2] + comb[row + 3] + hc_sinkhorn_eps;
    comb[row] /= denom;
    comb[row + 1] /= denom;
    comb[row + 2] /= denom;
    comb[row + 3] /= denom;
  }
  __syncthreads();

  if (tid < c) {
    const int col = tid;
    const float denom = comb[col] + comb[c + col] + comb[2 * c + col] + comb[3 * c + col] + hc_sinkhorn_eps;
    comb[col] /= denom;
    comb[c + col] /= denom;
    comb[2 * c + col] /= denom;
    comb[3 * c + col] /= denom;
  }
  __syncthreads();

  if (tid < c * c) comb_mix[token * c * c + tid] = comb[tid];

  for (int hidden = tid; hidden < h; hidden += blockDim.x) {
    const float r0 = mhc_bf16_to_float(residual[residual_base + hidden]);
    const float r1 = mhc_bf16_to_float(residual[residual_base + h + hidden]);
    const float r2 = mhc_bf16_to_float(residual[residual_base + 2 * h + hidden]);
    const float r3 = mhc_bf16_to_float(residual[residual_base + 3 * h + hidden]);
    const float out = ((pre[0] * r0 + pre[1] * r1) + (pre[2] * r2 + pre[3] * r3));
    layer_input[token * h + hidden] = mhc_float_to_bf16(out);
  }
}


extern "C" mhc_error_t mhc_pre_cuda_launch(
    const mhc_bfloat16* residual,
    const float* fn,
    const float* hc_scale,
    const float* hc_base,
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_mult_value,
    int sinkhorn_repeat,
    int t_tokens,
    int hc_mult,
    int hidden_size,
    float* post_mix,
    float* comb_mix,
    mhc_bfloat16* layer_input,
    mhc_stream_t stream) {
  const int threads = 256;
  if (hc_mult == 4 && hidden_size == 4096 && sinkhorn_repeat == 2) {
    const size_t shared_bytes = static_cast<size_t>(25 * threads + 24 + 4 + 16) * sizeof(float);
    mhc_pre_c4_h4096_kernel<<<t_tokens, threads, shared_bytes, stream>>>(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        t_tokens,
        post_mix,
        comb_mix,
        layer_input);
    return MHC_GET_LAST_ERROR();
  }

  const int c = hc_mult;
  const int c3 = c * 2 + c * c;
  const size_t shared_bytes =
      static_cast<size_t>(c3 + c + c * c + threads) * sizeof(float);

  mhc_pre_baseline_kernel<<<t_tokens, threads, shared_bytes, stream>>>(
      residual,
      fn,
      hc_scale,
      hc_base,
      rms_eps,
      hc_pre_eps,
      hc_sinkhorn_eps,
      hc_post_mult_value,
      sinkhorn_repeat,
      t_tokens,
      hc_mult,
      hidden_size,
      post_mix,
      comb_mix,
      layer_input);
  return MHC_GET_LAST_ERROR();
}
