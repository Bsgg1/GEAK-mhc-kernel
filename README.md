# GEAK MHC Pre Kernel Case

This repository contains a reproducible GEAK optimization case for the
`mhc_pre` operator on Hygon K500SM_AI DCU (`gfx928`).

It includes:

- The original CUDA/HIP baseline kernel.
- The GEAK-compatible example directory.
- The GEAK-optimized kernel.
- Correctness and benchmark harnesses.
- The final GEAK report and best patch.

## Repository Layout

```text
examples/mhc_ops/
  kernel.py                         # PyTorch reference/prototype operators
  test_mhc_ops_harness.py           # PyTorch reference harness
  src/mhc_pre.cu                    # CUDA/HIP mhc_pre baseline
  mhc_pre_hip_wrapper.py            # hipcc build + ctypes wrapper
  test_mhc_pre_hip_harness.py       # GEAK harness for the CUDA/HIP kernel
  config_mhc_pre_hip.yaml           # case config metadata

baseline/
  src/mhc_pre.cu                    # original baseline used before GEAK
  mhc_pre_hip_wrapper.py
  test_mhc_pre_hip_harness.py

optimized/
  src/mhc_pre.cu                    # GEAK best verified kernel
  mhc_pre_hip_wrapper.py            # GEAK best verified wrapper

results/
  final_report.json                 # GEAK final report
  best_round2_patch_1.patch         # best patch selected by GEAK

scripts/
  run-geak-mhc-pre.sh               # run GEAK on examples/mhc_ops
  test-baseline.sh                  # test baseline/
  test-optimized.sh                 # test optimized/
```

`baseline/` and `optimized/` are self-contained copies for comparison. The
`examples/mhc_ops/` tree is the GEAK-style example directory.

## Operator

`mhc_pre` receives a multi-channel residual stream and returns:

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

Validated environment:

- Hygon K500SM_AI DCU
- `gfx928`
- DTK/HIP with `hipcc`
- PyTorch with ROCm/DCU support
- GEAK installed separately when running optimization

The wrapper compiles the kernel with:

```bash
hipcc -x hip -O2 -shared -fPIC src/mhc_pre.cu \
  -o build/mhc_pre_hip/libmhc_pre.so \
  --offload-arch=gfx928
```

## Test Correctness

Test the optimized kernel:

```bash
bash scripts/test-optimized.sh --correctness
```

Expected output:

```text
hip correctness case 0: outer=(1,) C=4 H=128 ok
hip correctness case 1: outer=(3,) C=4 H=256 ok
hip correctness case 2: outer=(2, 3) C=4 H=512 ok
hip correctness case 3: outer=(2,) C=2 H=128 ok
hip correctness: ok
```

The harness compares the compiled kernel against an independent PyTorch
reference implementation. It checks:

- output dtype
- output shape
- numerical closeness

Tolerances:

- `post_mix`: `rtol=5e-4`, `atol=5e-4`
- `comb_mix`: `rtol=1e-3`, `atol=1e-3`
- `layer_input`: `rtol=3e-2`, `atol=3e-2`

`layer_input` is BF16, so it is compared after conversion to FP32.

## Test Performance

Quick benchmark:

```bash
bash scripts/test-optimized.sh --full-benchmark --iterations 5
```

Full benchmark:

```bash
bash scripts/test-optimized.sh --full-benchmark
```

The output contains per-shape latency and a GEAK metric:

```text
Perf: hip 0.110566 ms | outer=(1,) C=4 H=4096 sinkhorn_repeat=2
Perf: hip 0.116822 ms | outer=(16,) C=4 H=4096 sinkhorn_repeat=2
Perf: hip 0.135563 ms | outer=(64,) C=4 H=4096 sinkhorn_repeat=2
Perf: hip 0.176297 ms | outer=(128,) C=4 H=4096 sinkhorn_repeat=2
GEAK_RESULT_LATENCY_MS=0.132551
GEAK_RESULT_UNIT=ms
GEAK_RESULT_DIRECTION=lower_is_better
```

Lower `GEAK_RESULT_LATENCY_MS` is better.

## Test Baseline

Correctness:

```bash
bash scripts/test-baseline.sh --correctness
```

Benchmark:

```bash
bash scripts/test-baseline.sh --full-benchmark
```

## GEAK Result

Final verified result:

```text
baseline:  0.892604 ms
optimized: 0.132551 ms
speedup:   6.7340x
```

Both correctness and full benchmark passed.

Detailed report:

```bash
results/final_report.json
```

Best patch:

```bash
results/best_round2_patch_1.patch
```

## Run GEAK Again

This repository does not vendor the full GEAK framework. Install or clone GEAK
separately, then run this case by pointing GEAK at `examples/mhc_ops`.

Set `GEAK_ROOT` to your GEAK checkout if it is not `/root/GEAK`:

```bash
export GEAK_ROOT=/root/GEAK
```

Run:

```bash
bash scripts/run-geak-mhc-pre.sh
```

The script runs a command equivalent to:

```bash
geak --config "$GEAK_ROOT/config/local/hygon_k500sm_gfx928_codex_openai.yaml" \
  --repo "$PWD/examples/mhc_ops" \
  --kernel-url "$PWD/examples/mhc_ops/src/mhc_pre.cu" \
  --test-command "python3 $PWD/examples/mhc_ops/test_mhc_pre_hip_harness.py --full-benchmark" \
  --task "Optimize the CUDA/HIP mhc_pre baseline kernel for Hygon K500SM_AI DCU gfx928..." \
  --gpu-ids 0 \
  --num-parallel 1 \
  --debug \
  --yolo \
  --exit-immediately \
  -o "$GEAK_ROOT/optimization_logs/mhc_pre_cu_hygon_opt_from_repo"
```

The key pieces are:

- `--repo`: the case directory
- `--kernel-url`: the source file GEAK is allowed to modify
- `--test-command`: the harness command GEAK uses for correctness and benchmark
- `GEAK_RESULT_LATENCY_MS`: the metric parsed from harness output

## Notes

- Do not commit generated `build/`, `__pycache__/`, `.so`, or full
  `optimization_logs/` directories.
- The final GEAK profiler comparison was skipped because the profiler did not
  support `gfx928`; this does not affect correctness or full-benchmark
  verification.
