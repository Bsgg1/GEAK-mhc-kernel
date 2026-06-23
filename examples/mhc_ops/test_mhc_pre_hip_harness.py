#!/usr/bin/env python3
"""GEAK harness for the MHC pre CUDA/HIP baseline."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import torch

from mhc_pre_hip_wrapper import build_mhc_pre_library, mhc_pre_hip


RMS_EPS = 1.0e-6
HC_PRE_EPS = 1.0e-4
HC_SINKHORN_EPS = 1.0e-6
HC_POST_MULT_VALUE = 2.0
WARMUP = 5
DEFAULT_ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "20"))


@dataclass(frozen=True)
class Case:
    outer: tuple[int, ...]
    hc_mult: int
    hidden_size: int
    sinkhorn_repeat: int


CORRECTNESS_CASES = [
    Case((1,), 4, 128, 1),
    Case((3,), 4, 256, 2),
    Case((2, 3), 4, 512, 3),
    Case((2,), 2, 128, 2),
]

BENCHMARK_CASES = [
    Case((1,), 4, 4096, 2),
    Case((16,), 4, 4096, 2),
    Case((64,), 4, 4096, 2),
]

FULL_BENCHMARK_CASES = [
    Case((1,), 4, 4096, 2),
    Case((16,), 4, 4096, 2),
    Case((64,), 4, 4096, 2),
    Case((128,), 4, 4096, 2),
]


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("MHC HIP harness requires a CUDA/ROCm GPU")
    return torch.device("cuda")


def _make_inputs(case: Case, seed: int, device: torch.device):
    c = case.hc_mult
    h = case.hidden_size
    c3 = c * 2 + c * c
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    residual = torch.randn(
        (*case.outer, c, h),
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    fn = torch.randn(
        (c3, c * h),
        device=device,
        dtype=torch.float32,
        generator=gen,
    ) / math.sqrt(c * h)
    hc_scale = torch.tensor([0.75, 1.10, 0.50], device=device, dtype=torch.float32)
    hc_base = (
        torch.randn((c3,), device=device, dtype=torch.float32, generator=gen) * 0.05
    )
    return (
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        case.sinkhorn_repeat,
    )


def _mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == hc_scale.dtype == hc_base.dtype == torch.float32
    c = residual.shape[-2]
    h = residual.shape[-1]
    outer = residual.shape[:-2]
    r = residual.reshape(-1, c, h)
    t = r.shape[0]

    x = r.reshape(t, c * h).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (c * h) + rms_eps)

    pre_mix = torch.sigmoid(mixes[:, :c] * hc_scale[0] + hc_base[:c]) + hc_pre_eps
    post_mix = (
        torch.sigmoid(mixes[:, c : 2 * c] * hc_scale[1] + hc_base[c : 2 * c])
        * hc_post_mult_value
    )
    comb = (
        mixes[:, 2 * c :].reshape(t, c, c) * hc_scale[2]
        + hc_base[2 * c :].view(1, c, c)
    )
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(pre_mix.unsqueeze(-1) * r.to(torch.float32), dim=1).to(
        torch.bfloat16
    )
    return (
        post_mix.view(*outer, c, 1),
        comb.view(*outer, c, c),
        layer_input.view(*outer, h),
    )


def _geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("no benchmark samples")
    return math.exp(sum(math.log(max(v, 1.0e-12)) for v in values) / len(values))


def _print_metric(latency_ms: float) -> None:
    print(f"GEAK_RESULT_LATENCY_MS={latency_ms:.6f}")
    print("GEAK_RESULT_UNIT=ms")
    print("GEAK_RESULT_DIRECTION=lower_is_better")


def _run_correctness() -> int:
    build_mhc_pre_library(force=True)
    device = _device()
    for idx, case in enumerate(CORRECTNESS_CASES):
        inputs = _make_inputs(case, seed=5000 + idx, device=device)
        expected = _mhc_pre_reference(*inputs)
        actual = mhc_pre_hip(*inputs)

        if actual[0].dtype != torch.float32:
            raise AssertionError(f"post_mix dtype must be float32, got {actual[0].dtype}")
        if actual[1].dtype != torch.float32:
            raise AssertionError(f"comb_mix dtype must be float32, got {actual[1].dtype}")
        if actual[2].dtype != torch.bfloat16:
            raise AssertionError(
                f"layer_input dtype must be bfloat16, got {actual[2].dtype}"
            )
        for out, ref in zip(actual, expected):
            if out.shape != ref.shape:
                raise AssertionError(f"shape mismatch: got {out.shape}, expected {ref.shape}")

        torch.testing.assert_close(actual[0], expected[0], rtol=5.0e-4, atol=5.0e-4)
        torch.testing.assert_close(actual[1], expected[1], rtol=1.0e-3, atol=1.0e-3)
        torch.testing.assert_close(
            actual[2].float(),
            expected[2].float(),
            rtol=3.0e-2,
            atol=3.0e-2,
        )
        print(
            "hip correctness case "
            f"{idx}: outer={case.outer} C={case.hc_mult} H={case.hidden_size} ok"
        )
    print("hip correctness: ok")
    return 0


def _time_case(case: Case, iterations: int, seed: int, device: torch.device) -> float:
    inputs = _make_inputs(case, seed=6000 + seed, device=device)
    build_mhc_pre_library(force=True)

    for _ in range(WARMUP):
        mhc_pre_hip(*inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        mhc_pre_hip(*inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def _run_benchmark(cases: list[Case], iterations: int) -> int:
    device = _device()
    latencies: list[float] = []
    for idx, case in enumerate(cases):
        latency_ms = _time_case(case, iterations, seed=idx, device=device)
        latencies.append(latency_ms)
        print(
            "Perf: hip "
            f"{latency_ms:.6f} ms | outer={case.outer} C={case.hc_mult} "
            f"H={case.hidden_size} sinkhorn_repeat={case.sinkhorn_repeat}"
        )
    _print_metric(_geomean(latencies))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()

    iterations = max(1, args.iterations)
    if args.correctness:
        return _run_correctness()
    if args.benchmark:
        return _run_benchmark(BENCHMARK_CASES, iterations)
    if args.full_benchmark:
        return _run_benchmark(FULL_BENCHMARK_CASES, iterations)
    if args.profile:
        rc = _run_benchmark(BENCHMARK_CASES[:1], iterations)
        print("profile: hip event timing")
        return rc
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
