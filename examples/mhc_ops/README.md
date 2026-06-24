# MHC Operators

This example contains MHC operators and a GEAK-compatible harness.

Implemented:

- `mhc_pre` in `kernel.py`
- `mhc_post` in `kernel.py`
- `hc_head` in `kernel.py`
- CUDA baseline for `mhc_pre` in `src/mhc_pre.cu`
- CUDA baseline for `mhc_post` in `src/mhc_post.cu`
- CUDA baseline for `hc_head` in `src/mhc_head.cu`

The CUDA baseline is semantic-first and intentionally simple. On the current
Hygon/DTK machine it can be syntax-checked as HIP with:

```bash
/opt/dtk/bin/hipcc -x hip -c src/mhc_pre.cu -o /tmp/mhc_pre.o --offload-arch=gfx928
/opt/dtk/bin/hipcc -x hip -c src/mhc_post.cu -o /tmp/mhc_post.o --offload-arch=gfx928
/opt/dtk/bin/hipcc -x hip -c src/mhc_head.cu -o /tmp/mhc_head.o --offload-arch=gfx928
```

Run locally:

```bash
python3 test_mhc_ops_harness.py --correctness --operator all
python3 test_mhc_ops_harness.py --benchmark --operator pre
python3 test_mhc_ops_harness.py --benchmark --operator post
python3 test_mhc_ops_harness.py --benchmark --operator head
python3 test_mhc_ops_harness.py --full-benchmark --operator pre
python3 test_mhc_ops_harness.py --full-benchmark --operator post
python3 test_mhc_ops_harness.py --full-benchmark --operator head
python3 test_mhc_pre_hip_harness.py --correctness
python3 test_mhc_pre_hip_harness.py --full-benchmark
python3 test_mhc_post_hip_harness.py --correctness
python3 test_mhc_post_hip_harness.py --full-benchmark
python3 test_mhc_head_hip_harness.py --correctness
python3 test_mhc_head_hip_harness.py --full-benchmark
```

GEAK pre target:

```bash
geak --config config/local/hygon_k500sm_gfx928_codex_openai.yaml \
  --repo /root/GEAK/examples/mhc_ops \
  --kernel-url /root/GEAK/examples/mhc_ops/kernel.py \
  --test-command 'python3 /root/GEAK/examples/mhc_ops/test_mhc_ops_harness.py --full-benchmark --operator pre' \
  --task 'Optimize mhc_pre in /root/GEAK/examples/mhc_ops/kernel.py for Hygon K500SM_AI DCU gfx928. Metric is GEAK_RESULT_LATENCY_MS, lower is better. Preserve BF16/FP32 precision requirements and correctness for all harness shapes.' \
  --gpu-ids 0 \
  --num-parallel 1 \
  --debug \
  --yolo \
  --exit-immediately \
  -o /root/GEAK/optimization_logs/mhc_pre_hygon_opt
```

GEAK post target:

```bash
geak --config config/local/hygon_k500sm_gfx928_codex_openai.yaml \
  --repo /root/GEAK/examples/mhc_ops \
  --kernel-url /root/GEAK/examples/mhc_ops/kernel.py \
  --test-command 'python3 /root/GEAK/examples/mhc_ops/test_mhc_ops_harness.py --full-benchmark --operator post' \
  --task 'Optimize mhc_post in /root/GEAK/examples/mhc_ops/kernel.py for Hygon K500SM_AI DCU gfx928. Metric is GEAK_RESULT_LATENCY_MS, lower is better. Preserve the assumed post formula, FP32 accumulation, and BF16 output correctness for all harness shapes.' \
  --gpu-ids 0 \
  --num-parallel 1 \
  --debug \
  --yolo \
  --exit-immediately \
  -o /root/GEAK/optimization_logs/mhc_post_hygon_opt
```

GEAK post CUDA/HIP target:

```bash
geak --config config/local/hygon_k500sm_gfx928_codex_openai.yaml \
  --repo /root/GEAK/examples/mhc_ops \
  --kernel-url /root/GEAK/examples/mhc_ops/src/mhc_post.cu \
  --test-command 'python3 /root/GEAK/examples/mhc_ops/test_mhc_post_hip_harness.py --full-benchmark' \
  --task 'Optimize the CUDA/HIP mhc_post baseline kernel at /root/GEAK/examples/mhc_ops/src/mhc_post.cu for Hygon K500SM_AI DCU gfx928. The source is compiled by hipcc with -x hip. Metric is GEAK_RESULT_LATENCY_MS, lower is better. Preserve the formula new_residual_i = sum_j comb_mix[i,j] * residual_j + post_mix_i * x, FP32 accumulation, and BF16 output correctness for all harness shapes.' \
  --gpu-ids 0 \
  --num-parallel 1 \
  --debug \
  --yolo \
  --exit-immediately \
  -o /root/GEAK/optimization_logs/mhc_post_cu_hygon_opt
```

GEAK head CUDA/HIP target:

```bash
geak --config config/local/hygon_k500sm_gfx928_codex_openai.yaml \
  --repo /root/GEAK/examples/mhc_ops \
  --kernel-url /root/GEAK/examples/mhc_ops/src/mhc_head.cu \
  --test-command 'python3 /root/GEAK/examples/mhc_ops/test_mhc_head_hip_harness.py --full-benchmark' \
  --task 'Optimize the CUDA/HIP hc_head baseline kernel at /root/GEAK/examples/mhc_ops/src/mhc_head.cu for Hygon K500SM_AI DCU gfx928. The source is compiled by hipcc with -x hip. Metric is GEAK_RESULT_LATENCY_MS, lower is better. Preserve the formula hidden = sum_c head_mix[c] * residual[c], FP32 accumulation, and BF16 output correctness for all harness shapes.' \
  --gpu-ids 0 \
  --num-parallel 1 \
  --debug \
  --yolo \
  --exit-immediately \
  -o /root/GEAK/optimization_logs/mhc_head_cu_hygon_opt
```
