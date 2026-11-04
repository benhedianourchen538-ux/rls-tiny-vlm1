"""Shared helpers for all benchmarks: correct GPU timing and hardware recording."""
import platform
import subprocess
import time

import torch


def sync(device: torch.device):
    """GPU launches are asynchronous: without this, the timer stops when the work is QUEUED, not DONE."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_block(fn, device: torch.device, warmup: int, iters: int) -> float:
    """Mean seconds per call of fn(). Warm-up calls are discarded (CUDA context, cuDNN autotune,
    allocator growth, lazy kernel loading and CPU caches make the first iterations slower)."""
    for _ in range(warmup):
        fn()
    sync(device)                      # everything queued so far must be finished before the clock starts
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)                      # ... and finished before the clock stops
    return (time.perf_counter() - t0) / iters


def _run(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001 - best effort, hardware info must never crash a benchmark
        return f"unavailable ({e})"


def hardware_info() -> str:
    lines = [f"platform: {platform.platform()}", f"python: {platform.python_version()}",
             f"torch: {torch.__version__}", f"torch cuda: {torch.version.cuda}",
             f"cpu threads (torch): {torch.get_num_threads()}",
             "--- lscpu ---", _run("lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|MHz'")]
    if torch.cuda.is_available():
        lines += [f"gpu: {torch.cuda.get_device_name(0)}", "--- nvidia-smi ---",
                  _run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")]
    else:
        lines.append("gpu: none")
    return "\n".join(lines)
