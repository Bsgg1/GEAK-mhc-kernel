# GEAK MHC Kernel Case

This repository contains a reproducible GEAK optimization case for the
`mhc_pre`, `mhc_post`, and `hc_head` operators on Hygon K500SM_AI DCU
(`gfx928`).

It includes:

- The original CUDA/HIP baseline kernel.
- The GEAK-compatible example directory.
- The GEAK-optimized `mhc_pre` and `mhc_post` kernels.
- A validated `hc_head` baseline that is ready for GEAK optimization.
- Correctness and benchmark harnesses.
- The final GEAK report and best patch.

For a long-form Chinese write-up of the full Agent development process, see:

```text
docs/mhc_geak_agent_wechat_article.md
```

## Repository Layout

```text
examples/mhc_ops/
  kernel.py                         # PyTorch reference/prototype operators
  test_mhc_ops_harness.py           # PyTorch reference harness
  src/mhc_pre.cu                    # CUDA/HIP mhc_pre baseline
  src/mhc_post.cu                   # CUDA/HIP mhc_post baseline
  src/mhc_head.cu                   # CUDA/HIP hc_head baseline
  mhc_pre_hip_wrapper.py            # hipcc build + ctypes wrapper
  mhc_post_hip_wrapper.py           # hipcc build + ctypes wrapper
  mhc_head_hip_wrapper.py           # hipcc build + ctypes wrapper
  test_mhc_pre_hip_harness.py       # GEAK harness for the CUDA/HIP kernel
  test_mhc_post_hip_harness.py      # GEAK harness for the CUDA/HIP post kernel
  test_mhc_head_hip_harness.py      # GEAK harness for the CUDA/HIP head kernel
  config_mhc_pre_hip.yaml           # case config metadata
  config_mhc_post_hip.yaml          # post case config metadata
  config_mhc_head_hip.yaml          # head case config metadata

baseline/
  src/mhc_pre.cu                    # original baseline used before GEAK
  mhc_pre_hip_wrapper.py
  test_mhc_pre_hip_harness.py

optimized/
  src/mhc_pre.cu                    # GEAK best verified kernel
  mhc_pre_hip_wrapper.py            # GEAK best verified wrapper

optimized_post/
  src/mhc_post.cu                   # GEAK best verified post kernel
  mhc_post_hip_wrapper.py           # wrapper that builds optimized_post/src/mhc_post.cu

results/
  final_report.json                 # mhc_pre GEAK final report
  best_round2_patch_1.patch         # mhc_pre best patch selected by GEAK
  mhc_post_final_report.json        # mhc_post GEAK final report
  mhc_post_best_round2_patch_1.patch # mhc_post best patch selected by GEAK

scripts/
  run-geak-mhc-pre.sh               # run GEAK on examples/mhc_ops
  run-geak-mhc-post.sh              # run GEAK on the post kernel
  run-geak-mhc-head.sh              # run GEAK on the head kernel
  test-baseline.sh                  # test baseline/
  test-optimized.sh                 # test optimized/
  test-post-baseline.sh             # test examples/mhc_ops/src/mhc_post.cu
  test-post-optimized.sh            # test optimized_post/
  test-head-baseline.sh             # test examples/mhc_ops/src/mhc_head.cu
```

`baseline/` and `optimized/` are self-contained copies for comparison. The
`examples/mhc_ops/` tree is the GEAK-style example directory.

## Operator

`mhc_pre` receives a multi-channel residual stream and returns:

- `post_mix`: FP32, shape `[..., C, 1]`
- `comb_mix`: FP32, shape `[..., C, C]`
- `layer_input`: BF16, shape `[..., H]`

`mhc_post` consumes the sub-layer output and updates the multi-channel residual
stream. The assumed formula is:

```text
new_residual_i = sum_j comb_mix[i, j] * residual_j + post_mix_i * x
```

The post CUDA/HIP baseline is in:

```bash
examples/mhc_ops/src/mhc_post.cu
```

`hc_head` folds the final multi-channel residual stream back to a single
hidden stream before the LM head. The baseline formula is:

```text
hidden = sum_c head_mix[c] * residual[c]
```

The head CUDA/HIP baseline is in:

```bash
examples/mhc_ops/src/mhc_head.cu
```

Precision requirements:

- `residual`: BF16
- `fn`, `hc_scale`, `hc_base`: FP32
- GEMM/RMS/Sinkhorn intermediates: FP32
- `post_mix`, `comb_mix`: FP32
- `head_mix`: FP32
- `layer_input`, `new_residual`, `hidden`: BF16

## Agent Development Process

This case was developed with an Agent-assisted workflow instead of hand-tuning
only. The important point is that GEAK does not optimize a custom HIP kernel in
a vacuum: it needs a trusted baseline and a trusted harness first.

The development loop was:

```text
1. Inspect the target hardware and confirm Hygon K500SM_AI DCU / gfx928.
2. Prepare the GEAK local environment and OpenAI-compatible model config.
3. Write a semantic-first PyTorch reference for each operator.
4. Write a simple CUDA/HIP baseline kernel compiled by hipcc -x hip.
5. Build a Python ctypes wrapper that compiles the kernel into a shared object.
6. Build a harness with correctness, benchmark, full-benchmark, and profile modes.
7. Make the harness print GEAK_RESULT_LATENCY_MS for GEAK scoring.
8. Run correctness locally before starting GEAK.
9. Start GEAK with a precise prompt that includes hardware, metric, formula, and dtype requirements.
10. Inspect final_report.json, optimized_codes/, and the selected patch.
```

The prompt style that worked best was explicit and constraint-heavy. For
example, `mhc_post` was described as:

```text
Optimize the CUDA/HIP mhc_post baseline kernel for Hygon K500SM_AI DCU gfx928.
The source is compiled by hipcc with -x hip.
Metric is GEAK_RESULT_LATENCY_MS, lower is better.
Preserve the formula:
new_residual_i = sum_j comb_mix[i,j] * residual_j + post_mix_i * x.
Use FP32 accumulation and keep BF16 output correctness for all harness shapes.
```

Main issues encountered during development:

- A wrong API endpoint can return an HTML page instead of an LLM response. The
  model config must point at the OpenAI-compatible API path, and the key should
  be supplied through the environment rather than committed.
- Placeholder paths such as `/path/to/repo` cannot be used in GEAK commands.
  `--repo`, `--kernel-url`, and `--test-command` must use real absolute paths.
- Shell line breaks matter. Splitting `--full-benchmark` into `--full-` and
  `benchmark` makes Bash treat `benchmark` as a separate command.
- The harness is the contract. Agent-generated patches are only meaningful if
  correctness and benchmark checks are deterministic and representative.
- On this machine, profiler comparison for `gfx928` may be unsupported. That is
  separate from correctness and full-benchmark verification.

The correctness strategy is:

```text
1. Generate deterministic inputs for multiple shapes.
2. Compute an independent PyTorch reference.
3. Run the compiled HIP kernel.
4. Check output dtype and shape.
5. Compare numerically with tolerances appropriate for BF16 output.
6. Benchmark with torch.cuda.Event and report GEAK_RESULT_LATENCY_MS.
```

This workflow lets GEAK focus on the part it is good at: iterating over kernel
patches, testing candidates, and selecting the fastest verified implementation.

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

## Test Post Baseline

Correctness:

```bash
bash scripts/test-post-baseline.sh --correctness
```

Quick benchmark:

```bash
bash scripts/test-post-baseline.sh --full-benchmark --iterations 5
```

The post baseline harness compares the compiled CUDA/HIP kernel against an
independent PyTorch reference and reports `GEAK_RESULT_LATENCY_MS`.

## Test Head Baseline

Correctness:

```bash
bash scripts/test-head-baseline.sh --correctness
```

Quick benchmark:

```bash
bash scripts/test-head-baseline.sh --full-benchmark --iterations 5
```

The head baseline harness compares the compiled CUDA/HIP kernel against an
independent PyTorch reference and reports `GEAK_RESULT_LATENCY_MS`.

## Test Post Optimized

Correctness:

```bash
bash scripts/test-post-optimized.sh --correctness
```

Quick benchmark:

```bash
bash scripts/test-post-optimized.sh --full-benchmark --iterations 5
```

The optimized post result from GEAK is:

```text
baseline:  0.101829 ms
optimized: 0.089756 ms
speedup:   1.1345x
```

The optimized code is in:

```bash
optimized_post/src/mhc_post.cu
```

## GEAK Pre Result

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

## GEAK Post Result

Final verified result:

```text
baseline:  0.101829 ms
optimized: 0.089756 ms
speedup:   1.1345x
```

Both correctness and full benchmark passed.

Detailed report:

```bash
results/mhc_post_final_report.json
```

Best patch:

```bash
results/mhc_post_best_round2_patch_1.patch
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

To optimize the post CUDA/HIP baseline from `examples/mhc_ops/src/mhc_post.cu`:

```bash
bash scripts/run-geak-mhc-post.sh
```

To optimize the head CUDA/HIP baseline from `examples/mhc_ops/src/mhc_head.cu`:

```bash
bash scripts/run-geak-mhc-head.sh
```

The pre script runs a command equivalent to:

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
