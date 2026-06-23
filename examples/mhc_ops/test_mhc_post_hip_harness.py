#!/usr/bin/env python3
"""GEAK harness for the MHC post CUDA/HIP baseline."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import torch

from mhc_post_hip_wrapper import build_mhc_post_library, mhc_post_hip


HC_SINKHORN_EPS = 1.0e-6
WARMUP = 5
DEFAULT_ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "50"))


@dataclass(frozen=True)
class Case:
    outer: tuple[int, ...]
    hc_mult: int
    hidden_size: int


CORRECTNESS_CASES = [
    Case((1,), 4, 128),
    Case((3,), 4, 256),
    Case((2, 3), 4, 512),
    Case((2,), 2, 128),
]

BENCHMARK_CASES = [
    Case((1,), 4, 4096),
    Case((16,), 4, 4096),
    Case((64,), 4, 4096),
]

FULL_BENCHMARK_CASES = [
    Case((1,), 4, 4096),
    Case((16,), 4, 4096),
    Case((64,), 4, 4096),
    Case((128,), 4, 4096),
]


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("MHC HIP harness requires a CUDA/ROCm GPU")
    return torch.device("cuda")


def _sinkhorn_comb(logits: torch.Tensor) -> torch.Tensor:
    comb = torch.softmax(logits, dim=-1) + HC_SINKHORN_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_SINKHORN_EPS)
    comb = comb / (comb.sum(dim=-1, keepdim=True) + HC_SINKHORN_EPS)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_SINKHORN_EPS)
    return comb


def _make_inputs(case: Case, seed: int, device: torch.device):
    c = case.hc_mult
    h = case.hidden_size
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    x = torch.randn((*case.outer, h), device=device, dtype=torch.bfloat16, generator=gen)
    residual = torch.randn(
        (*case.outer, c, h),
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    post_mix = (
        torch.sigmoid(
            torch.randn((*case.outer, c, 1), device=device, dtype=torch.float32, generator=gen)
        )
        * 2.0
    )
    logits = torch.randn((*case.outer, c, c), device=device, dtype=torch.float32, generator=gen)
    comb_mix = _sinkhorn_comb(logits)
    return x, residual, post_mix, comb_mix


def _mhc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    assert x.dtype == residual.dtype == torch.bfloat16
    assert post_mix.dtype == comb_mix.dtype == torch.float32
    c = residual.shape[-2]
    h = residual.shape[-1]
    outer = residual.shape[:-2]
    r = residual.reshape(-1, c, h).to(torch.float32)
    y = x.reshape(-1, h).to(torch.float32)
    post = post_mix.reshape(-1, c, 1)
    comb = comb_mix.reshape(-1, c, c)
    out = torch.bmm(comb, r) + post * y.unsqueeze(1)
    return out.to(torch.bfloat16).view(*outer, c, h)


def _geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("no benchmark samples")
    return math.exp(sum(math.log(max(v, 1.0e-12)) for v in values) / len(values))


def _print_metric(latency_ms: float) -> None:
    print(f"GEAK_RESULT_LATENCY_MS={latency_ms:.6f}")
    print("GEAK_RESULT_UNIT=ms")
    print("GEAK_RESULT_DIRECTION=lower_is_better")


def _run_correctness() -> int:
    build_mhc_post_library(force=True)
    device = _device()
    for idx, case in enumerate(CORRECTNESS_CASES):
        inputs = _make_inputs(case, seed=7000 + idx, device=device)
        expected = _mhc_post_reference(*inputs)
        actual = mhc_post_hip(*inputs)

        if actual.dtype != torch.bfloat16:
            raise AssertionError(f"new_residual dtype must be bfloat16, got {actual.dtype}")
        if actual.shape != expected.shape:
            raise AssertionError(
                f"shape mismatch: got {actual.shape}, expected {expected.shape}"
            )

        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=3.0e-2,
            atol=3.0e-2,
        )
        flat_post_actual = mhc_post_hip(
            inputs[0],
            inputs[1],
            inputs[2].squeeze(-1),
            inputs[3],
        )
        torch.testing.assert_close(
            flat_post_actual.float(),
            expected.float(),
            rtol=3.0e-2,
            atol=3.0e-2,
        )
        print(
            "post hip correctness case "
            f"{idx}: outer={case.outer} C={case.hc_mult} H={case.hidden_size} ok"
        )
    print("post hip correctness: ok")
    return 0


def _time_case(case: Case, iterations: int, seed: int, device: torch.device) -> float:
    inputs = _make_inputs(case, seed=8000 + seed, device=device)
    build_mhc_post_library(force=True)

    for _ in range(WARMUP):
        mhc_post_hip(*inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        mhc_post_hip(*inputs)
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
            "Perf: post hip "
            f"{latency_ms:.6f} ms | outer={case.outer} C={case.hc_mult} "
            f"H={case.hidden_size}"
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
        print("profile: post hip event timing")
        return rc
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
