"""ctypes wrapper for the MHC post CUDA/HIP baseline."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src" / "mhc_post.cu"
BUILD_DIR = ROOT / "build" / "mhc_post_hip"
LIB_PATH = BUILD_DIR / "libmhc_post.so"


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


def build_mhc_post_library(force: bool = True) -> Path:
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
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "failed to build mhc_post HIP library\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return LIB_PATH


_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        lib_path = build_mhc_post_library(
            force=os.environ.get("MHC_POST_FORCE_REBUILD", "1") != "0"
        )
        lib = ctypes.CDLL(str(lib_path))
        lib.mhc_post_cuda_launch.argtypes = [
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # residual
            ctypes.c_void_p,  # post_mix
            ctypes.c_void_p,  # comb_mix
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,  # new_residual
            ctypes.c_void_p,  # stream
        ]
        lib.mhc_post_cuda_launch.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def mhc_post_hip(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    """Run the compiled MHC post baseline kernel."""

    if not x.is_cuda or not residual.is_cuda:
        raise ValueError("x and residual must be on CUDA/ROCm device")
    if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise ValueError("x and residual must be bfloat16")
    if post_mix.dtype != torch.float32 or comb_mix.dtype != torch.float32:
        raise ValueError("post_mix and comb_mix must be float32")

    c = residual.shape[-2]
    h = residual.shape[-1]
    outer = residual.shape[:-2]
    if x.shape != (*outer, h):
        raise ValueError(f"x must have shape {(*outer, h)}, got {tuple(x.shape)}")
    if comb_mix.shape != (*outer, c, c):
        raise ValueError(
            f"comb_mix must have shape {(*outer, c, c)}, got {tuple(comb_mix.shape)}"
        )
    if post_mix.shape == (*outer, c):
        post = post_mix.reshape(*outer, c, 1)
    elif post_mix.shape == (*outer, c, 1):
        post = post_mix
    else:
        raise ValueError(
            "post_mix must have shape "
            f"{(*outer, c)} or {(*outer, c, 1)}, got {tuple(post_mix.shape)}"
        )

    x = x.contiguous()
    residual = residual.contiguous()
    post = post.contiguous()
    comb_mix = comb_mix.contiguous()
    t_tokens = residual.numel() // (c * h)
    new_residual = torch.empty_like(residual)

    lib = _load_lib()
    torch.cuda.synchronize()
    err = lib.mhc_post_cuda_launch(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_void_p(post.data_ptr()),
        ctypes.c_void_p(comb_mix.data_ptr()),
        ctypes.c_int(int(t_tokens)),
        ctypes.c_int(int(c)),
        ctypes.c_int(int(h)),
        ctypes.c_void_p(new_residual.data_ptr()),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    if err != 0:
        raise RuntimeError(f"mhc_post_cuda_launch failed with HIP/CUDA error code {err}")
    return new_residual
