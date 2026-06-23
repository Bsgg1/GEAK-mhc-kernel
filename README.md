# GEAK MHC Pre Kernel Case

This repository contains a reproducible `mhc_pre` kernel optimization case from
GEAK on a Hygon K500SM_AI DCU (`gfx928`) machine.

## Contents

- `baseline/`: original semantic-first CUDA/HIP baseline and test harness.
- `optimized/`: GEAK-optimized kernel and wrapper.
- `results/final_report.json`: GEAK final verification report.
- `results/best_round2_patch_1.patch`: best patch selected by GEAK.
- `scripts/test-baseline.sh`: test the original baseline.
- `scripts/test-optimized.sh`: test the optimized kernel.

## Operator

`mhc_pre` consumes a multi-channel residual tensor and produces:

- `post_mix`: FP32, shape `[..., C, 1]`
- `comb_mix`: FP32, shape `[..., C, C]`
- `layer_input`: BF16, shape `[..., H]`

Precision requirements:

- `residual`: BF16
- `fn`, `hc_scale`, `hc_base`: FP32
- GEMM/RMS/Sinkhorn intermediates: FP32
- `post_mix`, `comb_mix`: FP32
- `layer_input`: BF16

## Requirements

This case was validated on:

- Hygon K500SM_AI DCU
- `gfx928`
- DTK/HIP with `hipcc`
- PyTorch with ROCm/DCU support

The kernel is compiled by the Python wrapper using:

```bash
hipcc -x hip -O2 -shared -fPIC src/mhc_pre.cu -o build/mhc_pre_hip/libmhc_pre.so --offload-arch=gfx928
```

## Correctness Test

Test the optimized kernel:

```bash
bash scripts/test-optimized.sh --correctness
```

Expected output includes:

```text
hip correctness case 0: outer=(1,) C=4 H=128 ok
hip correctness case 1: outer=(3,) C=4 H=256 ok
hip correctness case 2: outer=(2, 3) C=4 H=512 ok
hip correctness case 3: outer=(2,) C=2 H=128 ok
hip correctness: ok
```

The harness compares the compiled kernel against an independent PyTorch
reference implementation. It checks output dtype, output shape, and numerical
closeness:

- `post_mix`: `rtol=5e-4`, `atol=5e-4`
- `comb_mix`: `rtol=1e-3`, `atol=1e-3`
- `layer_input`: `rtol=3e-2`, `atol=3e-2`

`layer_input` is BF16, so it is compared after conversion to FP32 with a looser
tolerance.

## Benchmark

Quick benchmark:

```bash
bash scripts/test-optimized.sh --full-benchmark --iterations 5
```

Full benchmark:

```bash
bash scripts/test-optimized.sh --full-benchmark
```

The harness reports per-shape latency and a GEAK metric:

```text
GEAK_RESULT_LATENCY_MS=...
GEAK_RESULT_UNIT=ms
GEAK_RESULT_DIRECTION=lower_is_better
```

Lower `GEAK_RESULT_LATENCY_MS` is better.

## GEAK Result

Final verified result:

```text
baseline:  0.892604 ms
optimized: 0.132551 ms
speedup:   6.7340x
```

Correctness and full benchmark both passed.

See:

```bash
results/final_report.json
```

## Re-run Baseline

```bash
bash scripts/test-baseline.sh --correctness
bash scripts/test-baseline.sh --full-benchmark
```

## Re-run Optimized Kernel

```bash
bash scripts/test-optimized.sh --correctness
bash scripts/test-optimized.sh --full-benchmark
```
