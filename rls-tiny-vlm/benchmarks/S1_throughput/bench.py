"""S1 - training throughput (images/s) vs batch size, on CPU and GPU.

The three traps (Project Brief, section 9):
  1. warm-up      : first iterations are slower -> discarded (--warmup).
  2. async GPU    : torch.cuda.synchronize() before starting and before stopping the timer.
  3. data loading : EXCLUDED from the main measurement ("compute_only": one real batch kept on the device).
                    On GPU we also report "with_h2d": the uint8 batch is copied CPU->GPU inside the timer.
One "step" = forward + loss + backward + optimizer step. We report mean and std over --repeats.

    python -m benchmarks.S1_throughput.bench --device cpu
    python -m benchmarks.S1_throughput.bench --device cuda
"""
import argparse
import csv
from pathlib import Path

import torch

from benchmarks.common import hardware_info, time_block
from src.data import load_split, make_loader
from src.tokenizer import CharTokenizer
from src.train import build_model, compute_loss, load_config, make_optimizer, pick_device, set_seed

HERE = Path(__file__).parent


def make_step(model, opt, batch, device, with_h2d, pin):
    cpu_batch = {k: (v.pin_memory() if pin and torch.is_tensor(v) else v) for k, v in batch.items()}
    on_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def step():
        b = cpu_batch if with_h2d else on_dev          # with_h2d: the copy happens inside compute_loss
        loss = compute_loss(model, b, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 16, 64, 256])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--pin", action="store_true", help="pinned host memory for the h2d variant")
    a = p.parse_args()
    device = pick_device(a.device)
    cfg = load_config(a.config)
    set_seed(0)
    tok = CharTokenizer()
    ds = load_split(cfg["data_dir"], "train")
    (HERE / f"hardware_{device.type}.txt").write_text(hardware_info() + "\n")

    rows = []
    for bs in a.batch_sizes:
        batch = next(iter(make_loader(ds, tok, bs, shuffle=True, seed=0)))   # one REAL batch (realistic lengths)
        for mode in (["compute_only", "with_h2d"] if device.type == "cuda" else ["compute_only"]):
            model = build_model(cfg).to(device).train()
            opt = make_optimizer(model, 1e-3, 0.01)
            step = make_step(model, opt, batch, device, mode == "with_h2d", a.pin)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for r in range(a.repeats):
                sec = time_block(step, device, a.warmup if r == 0 else 1, a.iters)
                peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
                rows.append({"device": device.type, "mode": mode, "batch_size": bs, "repeat": r,
                             "ms_per_step": sec * 1e3, "images_per_s": bs / sec, "peak_mem_MiB": peak})
                print(rows[-1])
    csv_path = HERE / f"throughput_{device.type}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("raw timings ->", csv_path)


if __name__ == "__main__":
    main()
