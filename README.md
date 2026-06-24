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

## 中文使用说明：如何使用 GEAK Agent 调优 Kernel

这一节说明如何真正使用 GEAK Agent 做一次 kernel 调优。核心思想是：

```text
人负责定义问题边界：算子语义、输入输出、精度要求、正确性标准、benchmark 指标。
Agent 负责尝试优化实现：修改 kernel、运行 harness、比较指标、输出最优 patch。
```

也就是说，GEAK Agent 不是直接“凭空生成一个正确 kernel”，而是在已有 baseline
和 harness 的基础上进行自动调优。

### 1. 准备 GEAK 环境

先准备一个 GEAK checkout，并设置环境变量：

```bash
export GEAK_ROOT=/root/GEAK
export GEAK_CONFIG="$GEAK_ROOT/config/local/hygon_k500sm_gfx928_codex_openai.yaml"
export GEAK_USE_CODEX_OPENAI_KEY=1
```

如果 GEAK checkout 里有 Hygon DCU 环境脚本，可以加载：

```bash
source "$GEAK_ROOT/scripts/geak-hygon-env.sh"
```

这一步主要完成：

```text
1. 设置 Hygon K500SM_AI DCU / gfx928 相关环境
2. 设置 hipcc、Python、GEAK 路径
3. 设置 LLM API key 环境变量
4. 指定 GEAK 使用的模型配置
```

真实 API key 不应该写进仓库，建议通过环境变量传给 GEAK。

### 2. 选择要调优的 kernel

本仓库里有三个可作为 GEAK target 的 HIP/CUDA baseline：

```text
examples/mhc_ops/src/mhc_pre.cu
examples/mhc_ops/src/mhc_post.cu
examples/mhc_ops/src/mhc_head.cu
```

每个 target 都有对应 harness：

```text
examples/mhc_ops/test_mhc_pre_hip_harness.py
examples/mhc_ops/test_mhc_post_hip_harness.py
examples/mhc_ops/test_mhc_head_hip_harness.py
```

GEAK 调优前，需要先确认 baseline 本身是正确的。例如先测试 `hc_head`：

```bash
bash scripts/test-head-baseline.sh --correctness
bash scripts/test-head-baseline.sh --full-benchmark --iterations 5
```

如果 correctness 不通过，不要启动 GEAK。需要先修 baseline 或 reference。

### 3. Harness 必须提供什么

GEAK 依赖 harness 判断 patch 是否有效。一个合格 harness 至少要支持：

```text
--correctness       验证输出是否正确
--benchmark         快速性能测试
--full-benchmark    完整性能测试，GEAK 主要使用这个结果
--profile           可选，用于 profile
```

并且 benchmark 输出里必须包含：

```text
GEAK_RESULT_LATENCY_MS=0.108738
GEAK_RESULT_UNIT=ms
GEAK_RESULT_DIRECTION=lower_is_better
```

`GEAK_RESULT_LATENCY_MS` 是 GEAK 用来比较不同 patch 的指标。这里是 latency，所以
`lower_is_better`。

### 4. 启动 Agent 调优

最简单的方式是直接运行脚本：

```bash
bash scripts/run-geak-mhc-pre.sh
bash scripts/run-geak-mhc-post.sh
bash scripts/run-geak-mhc-head.sh
```

以 `hc_head` 为例，脚本内部等价于：

```bash
geak --config "$GEAK_CONFIG" \
  --repo "$PWD/examples/mhc_ops" \
  --kernel-url "$PWD/examples/mhc_ops/src/mhc_head.cu" \
  --test-command "python3 $PWD/examples/mhc_ops/test_mhc_head_hip_harness.py --full-benchmark" \
  --task "Optimize the CUDA/HIP hc_head baseline kernel for Hygon K500SM_AI DCU gfx928. The source is compiled by hipcc with -x hip. Metric is GEAK_RESULT_LATENCY_MS, lower is better. Preserve the formula hidden = sum_c head_mix[c] * residual[c], FP32 accumulation, and BF16 output correctness for all harness shapes." \
  --gpu-ids 0 \
  --num-parallel 1 \
  --debug \
  --yolo \
  --exit-immediately \
  -o "$GEAK_ROOT/optimization_logs/mhc_head_cu_hygon_opt_from_repo"
```

几个关键参数的作用：

```text
--config        GEAK 使用的模型和运行配置
--repo          被调优项目的根目录
--kernel-url    Agent 可以修改的 kernel 文件
--test-command  GEAK 用来验证 correctness 和 benchmark 的命令
--task          给 Agent 的自然语言任务说明
--gpu-ids       使用哪张 GPU
--num-parallel  同时跑几个子 Agent
--debug         保留完整日志和中间产物
--yolo          不进入交互确认，直接执行
-o              本次调优输出目录
```

### 5. Prompt 应该怎么写

调优效果很依赖 prompt。推荐包含以下信息：

```text
1. 目标硬件：Hygon K500SM_AI DCU gfx928
2. 源文件：具体 kernel 文件路径
3. 编译方式：hipcc -x hip
4. 优化目标：GEAK_RESULT_LATENCY_MS，lower is better
5. 数学公式：不能改变的算子语义
6. 精度要求：哪些输入是 BF16，哪些权重/中间量是 FP32
7. 正确性要求：必须通过所有 harness shapes
```

例如 `mhc_post` 的 prompt 可以写成：

```text
Optimize the CUDA/HIP mhc_post baseline kernel for Hygon K500SM_AI DCU gfx928.
The source is compiled by hipcc with -x hip.
Metric is GEAK_RESULT_LATENCY_MS, lower is better.
Preserve the formula:
new_residual_i = sum_j comb_mix[i,j] * residual_j + post_mix_i * x.
Use FP32 accumulation and keep BF16 output correctness for all harness shapes.
```

不要只写：

```text
Optimize this kernel.
```

这种 prompt 约束太弱，Agent 可能不知道哪些计算不能改。

### 6. 调优过程中怎么看日志

GEAK 输出目录通常在：

```text
$GEAK_ROOT/optimization_logs/<case_name>/
```

可以看主日志：

```bash
tail -f "$GEAK_ROOT/optimization_logs/mhc_head_cu_hygon_opt_from_repo/geak_agent.log"
```

重点关注几类信息：

```text
preprocess 是否通过
correctness 是否 PASS
full_benchmark 是否 PASS
Round N best 是哪个 patch
Verified speedup 是否大于 1
是否写出了 final_report.json
```

如果日志里出现：

```text
No valid candidates for evaluation
```

通常说明 Agent 生成的 patch 没有通过测试，或者没有成功提交候选结果。

### 7. 跑完后怎么看结果

一次成功的 GEAK run 通常会产生：

```text
final_report.json
optimized_codes/
results/round_*/fixed-canonical/patch_*.patch
round_*_evaluation.json
geak_agent.log
```

最重要的是：

```text
final_report.json          最终结果摘要
optimized_codes/           GEAK 选出的最终优化代码
patch_*.patch              对应的代码改动
round_*_evaluation.json    correctness / full_benchmark 验证细节
```

判断是否真的调优成功，看这几个字段：

```text
correctness.success = true
full_benchmark.success = true
verified_improvement = true
verified_speedup > 1.0
```

例如 `mhc_post` 的结果：

```text
baseline:  0.101829 ms
optimized: 0.089756 ms
speedup:   1.1345x
```

这表示 correctness 和 full benchmark 都通过，并且优化后 latency 更低。

### 8. 如何继续调优

如果一次 GEAK 结果不够好，可以从以下几个方向继续：

```text
1. 增加 --num-parallel，让多个 Agent 并行探索不同策略
2. 增加 benchmark iterations，降低计时噪声
3. 扩大 full-benchmark shapes，避免只优化单一 shape
4. 把上一次 best patch 应用为新 baseline，再跑下一轮
5. 在 prompt 中加入更明确的优化方向，例如向量化、减少全局内存访问、减少重复计算
6. 检查 harness 容差，避免过松或过严
7. 如果 profiler 支持目标架构，结合 profile 信息约束下一轮 prompt
```

常用环境变量：

```bash
export GEAK_BENCH_TIMEOUT=180
export GEAK_PROFILE_TIMEOUT=15
export GEAK_BASELINE_REPEATS=1
export GEAK_LLM_REQUEST_TIMEOUT=180
export MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=3
export GEAK_OFFLOAD_ARCH=gfx928
```

对于多 GPU 机器，可以尝试：

```bash
geak ... --gpu-ids 0,1,2,3 --num-parallel 4
```

这样 GEAK 会让多个子 Agent 在不同 GPU 上并行尝试优化。

### 9. 常见坑

```text
1. API base_url 配错，LLM 返回 HTML 页面，而不是模型响应。
2. OPENAI_API_KEY 旧值残留，导致实际用的不是当前 key。
3. --repo 或 --kernel-url 写成占位路径，GEAK 找不到文件。
4. shell 换行把 --full-benchmark 拆成了 --full- 和 benchmark。
5. harness 没有输出 GEAK_RESULT_LATENCY_MS，GEAK 无法比较性能。
6. baseline correctness 没过就启动 GEAK，后续 patch 结果没有意义。
7. benchmark iterations 太少，latency 抖动导致误判。
8. profile 工具不支持 gfx928，这不等价于 full_benchmark 失败。
```

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
