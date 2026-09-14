"""Device selection, environment capture, and an empirical FLOP roofline."""
from __future__ import annotations

import os
import platform
import random
import subprocess
import time

import numpy as np
import torch


def pick_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def device_memory_mb(device: torch.device) -> dict[str, float]:
    if device.type == "cuda":
        return dict(allocated_mb=torch.cuda.memory_allocated() / 1e6,
                    peak_mb=torch.cuda.max_memory_allocated() / 1e6)
    if device.type == "mps":
        return dict(allocated_mb=torch.mps.current_allocated_memory() / 1e6,
                    peak_mb=torch.mps.driver_allocated_memory() / 1e6)
    return dict(allocated_mb=0.0, peak_mb=0.0)


def _chip_name() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def env_info(device: torch.device | None = None) -> dict:
    d = device or pick_device()
    info = dict(
        python=platform.python_version(), torch=torch.__version__,
        platform=platform.platform(), chip=_chip_name(), device=str(d),
        cpu_count=os.cpu_count(),
    )
    if d.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    elif d.type == "mps":
        try:
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                       text=True, timeout=5).stdout.strip())
            info["unified_mem_gb"] = round(total / 1e9, 1)
        except Exception:
            pass
    return info


_PEAK_CACHE: dict[tuple, float] = {}


def measure_peak_flops(device: torch.device, dtype: torch.dtype = torch.bfloat16,
                       n: int = 2048, iters: int = 30) -> float:
    """Empirical dense-matmul roofline, in FLOP/s.

    MFU is normally quoted against a vendor peak number. On Apple silicon that
    number is both hard to pin down and not achievable by any real kernel, so we
    measure the machine's actual best-case matmul throughput and report MFU
    against that. It is a stricter, more honest denominator; we record it in the
    run's environment block so the number is interpretable later.
    """
    key = (str(device), str(dtype), n)
    if key in _PEAK_CACHE:
        return _PEAK_CACHE[key]
    if device.type == "cpu":
        n, iters = 1024, 8          # a 2048^3 matmul loop is far too slow on CPU
    try:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        for _ in range(5):
            _ = a @ b
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = a @ b
        sync(device)
        dt = time.perf_counter() - t0
        flops = iters * 2 * n ** 3 / dt
    except Exception:
        flops = float("nan")
    _PEAK_CACHE[key] = flops
    return flops


def human(n: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"
