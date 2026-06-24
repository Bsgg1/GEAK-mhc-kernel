# 用 Agent 在 Hygon DCU 上开发和优化 MHC Kernel 的一次完整实践

## 开篇

最近我们尝试把一个面向 DeepSeek V4 多通道残差流的 MHC 算子，放到 Hygon K500SM_AI DCU 上做实现和自动优化。这个过程不是简单地让 Agent 写一段 kernel 代码，而是从环境准备、baseline 实现、harness 搭建、Agent 提示词设计，到最后使用 GEAK 自动搜索优化版本的一整套流程。

这篇文章记录这次实践的完整开发过程，包括我们如何使用 Agent，如何准备 GEAK 运行环境，如何设计提示词，如何验证 kernel 正确性，以及中间踩到的一些坑。

## 背景：为什么要做 MHC Kernel

传统 Transformer 的残差流通常是一个单通道张量：

```text
hidden_states: [num_tokens, hidden_size]
```

DeepSeek V4 引入 Multi-Head Channel 机制后，残差流变成多通道结构：

```text
hidden_states: [num_tokens, hc_mult, hidden_size]
```

其中 `hc_mult` 通常为 4，每条通道承载不同的信息。Attention 和 FFN 子层本身仍然接收单通道输入，因此需要专门的 MHC 算子负责多通道和单通道之间的转换。

我们当前拆出来的几个核心算子是：

```text
mhc_pre              多通道 residual -> 单通道 layer_input
mhc_post             子层输出 + residual -> 新的多通道 residual
mhc_fused_post_pre   融合当前 post 和下一个 pre
hc_head              最后将多通道 residual 折叠回单通道，送入 LM Head
```

这次实践中，我们先完成了 `mhc_pre`、`mhc_post`，随后补充了 `hc_head` baseline。

## 硬件和软件环境

目标硬件是 Hygon K500SM_AI DCU，架构为 `gfx928`。机器上有 4 张 DCU 卡，每张卡约 64GB 显存。

通过 `lspci` 可以看到设备类似：

```text
Co-processor: Chengdu C-3000 IC Design Co., Ltd. KONGMING
```

GEAK 侧使用的核心环境包括：

```text
GEAK_ROOT=/root/GEAK
GEAK_GPU_IDS=0,1,2,3
GEAK_TARGET_HARDWARE=Hygon K500SM_AI DCU, gfx928
GEAK_OFFLOAD_ARCH=gfx928
```

编译 HIP kernel 时使用：

```bash
hipcc -x hip --offload-arch=gfx928
```

这里有一个关键点：虽然源码文件后缀是 `.cu`，但在 Hygon DCU 上实际是通过 `hipcc -x hip` 编译为 HIP 代码。这样做的好处是代码可以保持 CUDA/HIP 双端结构，同时在当前机器上以 HIP 方式编译运行。

## Agent 环境准备

这次使用的是 GEAK，它基于 Mini-SWE-Agent 风格的流程，负责调度子 Agent 修改代码、运行测试、收集 patch、评估性能并选择最优结果。

我们增加了一个本地环境脚本：

```bash
source scripts/geak-hygon-env.sh
```

这个脚本主要做几件事：

```text
1. 设置 GEAK_ROOT、GPU ID、目标硬件信息
2. 配置 PATH 和 PYTHONPATH
3. 设置 gfx928 编译目标
4. 在需要时从本机安全位置读取 OPENAI_API_KEY
```

LLM 配置放在：

```text
config/local/hygon_k500sm_gfx928_codex_openai.yaml
```

其中只保存模型名和 OpenAI-compatible `base_url`，不把真实 key 写进仓库。key 通过环境变量 `OPENAI_API_KEY` 注入。

启动前一般执行：

```bash
cd /root/GEAK
unset OPENAI_API_KEY
export GEAK_USE_CODEX_OPENAI_KEY=1
source scripts/geak-hygon-env.sh
```

这里 `unset OPENAI_API_KEY` 是为了避免 shell 里残留旧 key，导致 GEAK 实际用的不是最新配置。

## GEAK 需要哪些输入

GEAK 不是凭空优化 kernel。对这种自定义算子，我们必须先准备好三样东西：

```text
1. 一个能工作的 baseline kernel
2. 一个可以验证正确性的 harness
3. 一个可以输出性能指标的 benchmark
```

harness 的输出必须包含 GEAK 能解析的指标：

```text
GEAK_RESULT_LATENCY_MS=0.132551
GEAK_RESULT_UNIT=ms
GEAK_RESULT_DIRECTION=lower_is_better
```

其中 `GEAK_RESULT_LATENCY_MS` 是最终用于比较 patch 优劣的关键指标。

这点非常重要。Agent 可以帮我们改 kernel，但不能可靠地替我们发明正确的业务语义。比如 MHC 算子的 dtype、shape、FP32 累加要求、BF16 输出要求、Sinkhorn 细节，都必须由我们在 reference 和 harness 里定义清楚。

## Harness 是怎么设计的

以 `mhc_pre` 为例，harness 做了几件事：

```text
1. 构造多组输入 shape
2. 用 PyTorch reference 计算期望结果
3. 编译并调用 HIP kernel
4. 检查输出 shape 和 dtype
5. 用 torch.testing.assert_close 比较数值
6. 用 torch.cuda.Event 计时
7. 打印 GEAK_RESULT_LATENCY_MS
```

`mhc_pre` 的输出包括：

```text
post_mix:    FP32, shape [..., C, 1]
comb_mix:    FP32, shape [..., C, C]
layer_input: BF16, shape [..., H]
```

由于 `layer_input` 是 BF16，比较时先转为 FP32，然后使用相对宽松的容差：

```text
post_mix:    rtol=5e-4, atol=5e-4
comb_mix:    rtol=1e-3, atol=1e-3
layer_input: rtol=3e-2, atol=3e-2
```

`mhc_post` 的 reference 采用如下公式：

```text
new_residual_i = sum_j comb_mix[i, j] * residual_j + post_mix_i * x
```

`hc_head` 的 baseline 当前采用如下折叠公式：

```text
hidden = sum_c head_mix[c] * residual[c]
```

所有 kernel 都遵循同样原则：

```text
输入 residual 使用 BF16
权重或混合系数使用 FP32
中间累加使用 FP32
最终输出按算子定义转回 BF16
```

## Baseline 目录结构

MHC 相关代码放在：

```text
examples/mhc_ops/
```

主要文件包括：

```text
kernel.py                         PyTorch reference
src/mhc_pre.cu                    mhc_pre HIP baseline
src/mhc_post.cu                   mhc_post HIP baseline
src/mhc_head.cu                   hc_head HIP baseline
mhc_pre_hip_wrapper.py            ctypes wrapper
mhc_post_hip_wrapper.py           ctypes wrapper
mhc_head_hip_wrapper.py           ctypes wrapper
test_mhc_pre_hip_harness.py       mhc_pre harness
test_mhc_post_hip_harness.py      mhc_post harness
test_mhc_head_hip_harness.py      hc_head harness
```

这些 wrapper 负责调用 `hipcc` 生成 `.so`，再通过 `ctypes` 从 Python 中调用 kernel。

## 提示词准备

GEAK 优化时，提示词需要尽量具体。不能只写“帮我优化这个 kernel”，而要告诉 Agent：

```text
1. 目标硬件是什么
2. 源文件是什么
3. 编译方式是什么
4. metric 是什么
5. 正确性约束是什么
6. 哪些精度不能改变
```

以 `mhc_pre` 为例，提示词类似：

```text
Optimize the CUDA/HIP mhc_pre baseline kernel at
/root/GEAK/examples/mhc_ops/src/mhc_pre.cu for Hygon K500SM_AI DCU gfx928.
The source is compiled by hipcc with -x hip.
Metric is GEAK_RESULT_LATENCY_MS, lower is better.
Preserve correctness for all harness shapes.
Precision requirements: residual is BF16, fn/hc_scale/hc_base are FP32,
GEMM accumulation is FP32, post_mix and comb_mix are FP32,
and layer_input output is BF16.
```

`mhc_post` 的提示词会明确公式：

```text
Preserve the formula
new_residual_i = sum_j comb_mix[i,j] * residual_j + post_mix_i * x,
FP32 accumulation, and BF16 output correctness for all harness shapes.
```

`hc_head` 的提示词则明确折叠公式：

```text
Preserve the formula
hidden = sum_c head_mix[c] * residual[c],
FP32 accumulation, and BF16 output correctness for all harness shapes.
```

## 实际启动 GEAK

`mhc_pre` 的优化命令结构如下：

```bash
geak --config config/local/hygon_k500sm_gfx928_codex_openai.yaml \
  --repo /root/GEAK/examples/mhc_ops \
  --kernel-url /root/GEAK/examples/mhc_ops/src/mhc_pre.cu \
  --test-command 'python3 /root/GEAK/examples/mhc_ops/test_mhc_pre_hip_harness.py --full-benchmark' \
  --task 'Optimize the CUDA/HIP mhc_pre baseline kernel at /root/GEAK/examples/mhc_ops/src/mhc_pre.cu for Hygon K500SM_AI DCU gfx928. The source is compiled by hipcc with -x hip. Metric is GEAK_RESULT_LATENCY_MS, lower is better. Preserve correctness for all harness shapes. Precision requirements: residual is BF16, fn/hc_scale/hc_base are FP32, GEMM accumulation is FP32, post_mix and comb_mix are FP32, and layer_input output is BF16.' \
  --gpu-ids 0 \
  --num-parallel 1 \
  --debug \
  --yolo \
  --exit-immediately \
  -o /root/GEAK/optimization_logs/mhc_pre_cu_hygon_opt_agent
```

几个参数的含义：

```text
--config        GEAK 的模型和运行配置
--repo          被优化项目的根目录
--kernel-url    允许 Agent 修改的 kernel 文件
--test-command  用于 correctness 和 benchmark 的 harness 命令
--task          给 Agent 的优化任务描述
--gpu-ids       使用哪张 GPU
--num-parallel  并行子 Agent 数量
--debug         输出更多日志
--yolo          跳过交互确认，直接执行
-o              输出日志和产物目录
```

优化结束后，重点看：

```text
final_report.json
optimized_codes/
results/round_x/.../patch_*.patch
```

GEAK 会把最终验证通过的代码复制到 `optimized_codes/`。

## 本次优化结果

`mhc_pre` 的 GEAK 优化结果：

```text
baseline:  0.892604 ms
optimized: 0.132551 ms
speedup:   6.7340x
```

`mhc_post` 的 GEAK 优化结果：

```text
baseline:  0.101829 ms
optimized: 0.089756 ms
speedup:   1.1345x
```

`hc_head` 目前已经完成 baseline 和 harness，短 benchmark 能正常输出：

```text
GEAK_RESULT_LATENCY_MS=0.108738
```

下一步可以对 `hc_head` 继续跑 GEAK 优化。

## 中间遇到的坑

### 1. API base_url 配错会返回 HTML

一开始 LiteLLM 报错：

```text
Empty or invalid response from LLM endpoint
Received: <!doctype html>...
```

这说明请求打到了网关前端页面，而不是 OpenAI-compatible API endpoint。最后确认 GEAK 配置里应该使用 `/v1` API 路径，同时 key 通过环境变量注入。

### 2. 不能把 placeholder 路径直接拿去跑

早期命令里出现了：

```text
/path/to/repo
/path/to/kernel.py
```

GEAK 会直接按真实路径解析，结果报：

```text
--repo directory not found: /path/to/repo
```

所以命令里必须写真实 repo 和 kernel 文件路径。

### 3. shell 换行错误会把参数拆坏

有一次命令写成：

```bash
--full-
benchmark
```

shell 会把它当成两个 token，甚至把 `benchmark` 当成命令执行，出现：

```text
bash: benchmark: command not found
```

解决方式是：长命令每一行都用反斜杠结尾，参数本身不要拆开。

### 4. GEAK 需要 baseline 和 harness

GEAK 不是万能代码生成器。对于自定义 HIP kernel，必须先有人定义清楚 reference 和 correctness。否则 Agent 即使写出更快的代码，也无法判断是不是正确。

这也是为什么我们先写：

```text
src/mhc_pre.cu
test_mhc_pre_hip_harness.py
```

再启动 GEAK。

### 5. 非 git worktree 会影响 postprocess

GEAK 在评估 patch 时会创建临时 worktree，并执行 git commit。如果机器没有配置 git identity，可能报：

```text
git commit returned non-zero exit status 128
```

我们在 GEAK postprocess 里补了默认 identity：

```text
GEAK Agent <geak@amd.com>
```

这样非 git 示例目录也能正常评估。

### 6. gfx928 profiler 不一定支持

GEAK 的 profiler 阶段曾提示：

```text
Unsupported architecture: gfx928
```

这不会影响 correctness 和 full benchmark。最终选择 patch 时，GEAK 仍然以 harness 的 `FULL_BENCHMARK` 验证结果为准。

## 如何验证优化后的代码

以 `mhc_pre` 为例，优化后的代码在：

```text
optimization_logs/mhc_pre_cu_hygon_opt_agent/optimized_codes/
```

可以通过脚本测试：

```bash
bash scripts/test-mhc-pre-optimized.sh --correctness
bash scripts/test-mhc-pre-optimized.sh --full-benchmark --iterations 5
```

`mhc_post` 的优化结果可以在：

```text
optimization_logs/mhc_post_cu_hygon_opt/optimized_codes/
```

中查看。

判断结果是否有效，主要看三点：

```text
1. correctness 是否 PASS
2. full_benchmark 是否 PASS
3. verified speedup 是否大于 1
```

## 对外复现建议

如果要把这个 case 给其他同事或社区复现，建议提供以下内容：

```text
1. baseline kernel
2. PyTorch reference
3. harness
4. GEAK 启动脚本
5. final_report.json
6. best patch
7. README 中写清楚环境和命令
```

我们已经把 MHC case 整理成单独仓库，包含：

```text
examples/mhc_ops/
baseline/
optimized/
optimized_post/
results/
scripts/
```

这样别人不需要理解 GEAK 内部所有实现，也可以先跑 correctness 和 benchmark，再决定是否接入 GEAK 重新优化。

## 小结

这次实践最大的经验是：Agent 在 kernel 优化场景里很有价值，但前提是工程边界要清楚。

我们需要人来定义：

```text
算子语义
输入输出 shape
精度要求
reference 实现
正确性判定
性能指标
```

Agent 更适合承担：

```text
阅读现有代码
生成优化 patch
反复运行 harness
根据 benchmark 结果迭代
整理最终优化产物
```

也就是说，Agent 不是替代工程判断，而是把“实现、尝试、验证、筛选”这部分循环自动化。对于 MHC 这类形状固定、语义明确、性能敏感的 kernel，这种工作流非常适合。

下一步可以继续优化 `hc_head`，再做 `mhc_fused_post_pre`，把 Attention 后的 post 和 FFN 前的 pre 融合起来，减少中间 residual 的读写开销。
