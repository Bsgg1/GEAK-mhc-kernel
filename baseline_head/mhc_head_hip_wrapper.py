"""ctypes wrapper for the MHC head CUDA/HIP baseline."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src" / "mhc_head.cu"
BUILD_DIR = ROOT / "build" / "mhc_head_hip"
LIB_PATH = BUILD_DIR / "libmhc_head.so"


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


def build_mhc_head_library(force: bool = True) -> Path:
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
            "failed to build mhc_head HIP library\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return LIB_PATH


_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        lib_path = build_mhc_head_library(
            force=os.environ.get("MHC_HEAD_FORCE_REBUILD", "1") != "0"
        )
        lib = ctypes.CDLL(str(lib_path))
        lib.hc_head_cuda_launch.argtypes = [
            ctypes.c_void_p,  # residual
            ctypes.c_void_p,  # head_mix
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,  # hidden
            ctypes.c_void_p,  # stream
        ]
        lib.hc_head_cuda_launch.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def hc_head_hip(residual: torch.Tensor, head_mix: torch.Tensor) -> torch.Tensor:
    """Run the compiled MHC head baseline kernel."""

    if not residual.is_cuda:
        raise ValueError("residual must be on CUDA/ROCm device")
    if residual.dtype != torch.bfloat16:
        raise ValueError(f"residual must be bfloat16, got {residual.dtype}")
    if head_mix.dtype != torch.float32:
        raise ValueError(f"head_mix must be float32, got {head_mix.dtype}")

    c = residual.shape[-2]
    h = residual.shape[-1]
    outer = residual.shape[:-2]
    t_tokens = residual.numel() // (c * h)

    if head_mix.shape == (c,):
        head = head_mix.reshape(1, c).expand(t_tokens, c).contiguous()
    elif head_mix.shape == (*outer, c):
        head = head_mix.reshape(t_tokens, c).contiguous()
    else:
        raise ValueError(
            f"head_mix must have shape {(c,)} or {(*outer, c)}, "
            f"got {tuple(head_mix.shape)}"
        )

    residual = residual.contiguous()
    hidden = torch.empty((*outer, h), device=residual.device, dtype=torch.bfloat16)

    lib = _load_lib()
    err = lib.hc_head_cuda_launch(
        ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_void_p(head.data_ptr()),
        ctypes.c_int(int(t_tokens)),
        ctypes.c_int(int(c)),
        ctypes.c_int(int(h)),
        ctypes.c_void_p(hidden.data_ptr()),
        ctypes.c_void_p(0),
    )
    if err != 0:
        raise RuntimeError(f"hc_head_cuda_launch failed with HIP/CUDA error code {err}")
    return hidden
