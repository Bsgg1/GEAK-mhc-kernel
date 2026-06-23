# MHC Operators

This example contains MHC operators and a GEAK-compatible harness.

Implemented:

- `mhc_pre` in `kernel.py`
- `mhc_post` in `kernel.py`
- CUDA baseline for `mhc_pre` in `src/mhc_pre.cu`

The CUDA baseline is semantic-first and intentionally simple. On the current
Hygon/DTK machine it can be syntax-checked as HIP with:

```bash
/opt/dtk/bin/hipcc -x hip -c src/mhc_pre.cu -o /tmp/mhc_pre.o --offload-arch=gfx928
```

Run locally:

```bash
python3 test_mhc_ops_harness.py --correctness --operator all
python3 test_mhc_ops_harness.py --benchmark --operator pre
python3 test_mhc_ops_harness.py --benchmark --operator post
python3 test_mhc_ops_harness.py --full-benchmark --operator pre
python3 test_mhc_ops_harness.py --full-benchmark --operator post
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
