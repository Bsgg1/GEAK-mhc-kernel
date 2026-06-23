"""ctypes wrapper for the MHC pre CUDA/HIP baseline."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src" / "mhc_pre.cu"
BUILD_DIR = ROOT / "build" / "mhc_pre_hip"
LIB_PATH = BUILD_DIR / "libmhc_pre.so"


def _hipcc() -> str:
    candidates = [
        os.environ.get("HIPCC"),
        shutil.which("hipcc"),
        "/opt/dtk/bin/hipcc",
        "/opt/rocm/bin/hipcc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("hipcc not found; set HIPCC or source the DTK/ROCm environment")


def build_mhc_pre_library(force: bool = True) -> Path:
    """Build the shared library used by the Python harness."""

    if not SRC.exists():
        raise FileNotFoundError(SRC)
    if not force and LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= SRC.stat().st_mtime:
        return LIB_PATH

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    arch = os.environ.get("GEAK_OFFLOAD_ARCH", "gfx928")
    cmd = [
        _hipcc(),
        "-x",
        "hip",
        "-O2",
        "-shared",
        "-fPIC",
        str(SRC),
        "-o",
        str(LIB_PATH),
        f"--offload-arch={arch}",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "failed to build mhc_pre HIP library\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return LIB_PATH


_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        lib_path = build_mhc_pre_library(force=os.environ.get("MHC_PRE_FORCE_REBUILD", "1") != "0")
        lib = ctypes.CDLL(str(lib_path))
        lib.mhc_pre_cuda_launch.argtypes = [
            ctypes.c_void_p,  # residual
            ctypes.c_void_p,  # fn
            ctypes.c_void_p,  # hc_scale
            ctypes.c_void_p,  # hc_base
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,  # post_mix
            ctypes.c_void_p,  # comb_mix
            ctypes.c_void_p,  # layer_input
            ctypes.c_void_p,  # stream
        ]
        lib.mhc_pre_cuda_launch.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def mhc_pre_hip(
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
    """Run the compiled MHC pre baseline kernel."""

    if not residual.is_cuda:
        raise ValueError("residual must be on CUDA/ROCm device")
    if residual.dtype != torch.bfloat16:
        raise ValueError(f"residual must be bfloat16, got {residual.dtype}")
    if fn.dtype != torch.float32 or hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32:
        raise ValueError("fn, hc_scale, and hc_base must be float32")

    residual = residual.contiguous()
    fn = fn.contiguous()
    hc_scale = hc_scale.contiguous()
    hc_base = hc_base.contiguous()

    c = residual.shape[-2]
    h = residual.shape[-1]
    c3 = c * 2 + c * c
    outer = residual.shape[:-2]
    t_tokens = residual.numel() // (c * h)
    if fn.shape != (c3, c * h):
        raise ValueError(f"fn must have shape {(c3, c * h)}, got {tuple(fn.shape)}")
    if hc_base.shape != (c3,):
        raise ValueError(f"hc_base must have shape {(c3,)}, got {tuple(hc_base.shape)}")
    if hc_scale.numel() < 3:
        raise ValueError("hc_scale must contain at least 3 values")

    post_mix = torch.empty((*outer, c, 1), device=residual.device, dtype=torch.float32)
    comb_mix = torch.empty((*outer, c, c), device=residual.device, dtype=torch.float32)
    layer_input = torch.empty((*outer, h), device=residual.device, dtype=torch.bfloat16)

    lib = _load_lib()
    torch.cuda.synchronize()
    err = lib.mhc_pre_cuda_launch(
        ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_void_p(fn.data_ptr()),
        ctypes.c_void_p(hc_scale.data_ptr()),
        ctypes.c_void_p(hc_base.data_ptr()),
        ctypes.c_float(float(rms_eps)),
        ctypes.c_float(float(hc_pre_eps)),
        ctypes.c_float(float(hc_sinkhorn_eps)),
        ctypes.c_float(float(hc_post_mult_value)),
        ctypes.c_int(int(sinkhorn_repeat)),
        ctypes.c_int(int(t_tokens)),
        ctypes.c_int(int(c)),
        ctypes.c_int(int(h)),
        ctypes.c_void_p(post_mix.data_ptr()),
        ctypes.c_void_p(comb_mix.data_ptr()),
        ctypes.c_void_p(layer_input.data_ptr()),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    if err != 0:
        raise RuntimeError(f"mhc_pre_cuda_launch failed with HIP/CUDA error code {err}")
    return post_mix, comb_mix, layer_input
