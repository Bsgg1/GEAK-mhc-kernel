#!/usr/bin/env python3
"""GEAK harness for the MHC head CUDA/HIP baseline."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import torch

from mhc_head_hip_wrapper import build_mhc_head_library, hc_head_hip


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


def _make_inputs(case: Case, seed: int, device: torch.device):
    c = case.hc_mult
    h = case.hidden_size
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    residual = torch.randn(
        (*case.outer, c, h),
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    logits = torch.randn((*case.outer, c), device=device, dtype=torch.float32, generator=gen)
    head_mix = torch.softmax(logits, dim=-1)
    return residual, head_mix


def _hc_head_reference(residual: torch.Tensor, head_mix: torch.Tensor) -> torch.Tensor:
    assert residual.dtype == torch.bfloat16
    assert head_mix.dtype == torch.float32
    c = residual.shape[-2]
    h = residual.shape[-1]
    outer = residual.shape[:-2]
    r = residual.reshape(-1, c, h).to(torch.float32)
    if head_mix.shape == (c,):
        head = head_mix.view(1, c).expand(r.shape[0], c)
    else:
        head = head_mix.reshape(-1, c)
    hidden = torch.sum(head.unsqueeze(-1) * r, dim=1)
    return hidden.to(torch.bfloat16).view(*outer, h)


def _geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("no benchmark samples")
    return math.exp(sum(math.log(max(v, 1.0e-12)) for v in values) / len(values))


def _print_metric(latency_ms: float) -> None:
    print(f"GEAK_RESULT_LATENCY_MS={latency_ms:.6f}")
    print("GEAK_RESULT_UNIT=ms")
    print("GEAK_RESULT_DIRECTION=lower_is_better")


def _run_correctness() -> int:
    build_mhc_head_library(force=True)
    device = _device()
    for idx, case in enumerate(CORRECTNESS_CASES):
        inputs = _make_inputs(case, seed=11000 + idx, device=device)
        expected = _hc_head_reference(*inputs)
        actual = hc_head_hip(*inputs)

        if actual.dtype != torch.bfloat16:
            raise AssertionError(f"hidden dtype must be bfloat16, got {actual.dtype}")
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

        flat_head = inputs[1].reshape(-1, case.hc_mult)[0]
        flat_expected = _hc_head_reference(inputs[0], flat_head)
        flat_actual = hc_head_hip(inputs[0], flat_head)
        torch.testing.assert_close(
            flat_actual.float(),
            flat_expected.float(),
            rtol=3.0e-2,
            atol=3.0e-2,
        )
        print(
            "head hip correctness case "
            f"{idx}: outer={case.outer} C={case.hc_mult} H={case.hidden_size} ok"
        )
    print("head hip correctness: ok")
    return 0


def _time_case(case: Case, iterations: int, seed: int, device: torch.device) -> float:
    inputs = _make_inputs(case, seed=12000 + seed, device=device)
    build_mhc_head_library(force=True)

    for _ in range(WARMUP):
        hc_head_hip(*inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        hc_head_hip(*inputs)
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
            "Perf: head hip "
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
        print("profile: head hip event timing")
        return rc
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
