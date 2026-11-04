"""Level 2 - peak GPU memory vs batch size: PREDICT first, then measure (CUDA only).

Peak memory of a training step = parameters + gradients + optimizer state (AdamW: 2 buffers) + activations
saved for backward (+ transient workspace). Only the activation term grows with the batch size.

Activation estimate (float32, per sample, T = V + L tokens, D = d_model, H heads, N layers):
  per decoder block, tensors kept for backward ~ 16*T*D  (LN outputs, qkv, attn output, MLP 4D hidden x2, residuals)
                                                  + 3*H*T*T (scores, softmax weights, dropout output)
  CNN: for each conv block: conv out + BN out + ReLU out (3 x C x H x W), at the resolution of that block.
The estimate is rough on purpose: the point is the ORDER OF MAGNITUDE and which term dominates.

    python -m benchmarks.level2.memory --device cuda
"""
import argparse
import csv
from pathlib import Path

import torch

from benchmarks.common import hardware_info
from src.data import load_split, make_loader
from src.tokenizer import CharTokenizer
from src.train import build_model, compute_loss, load_config, make_optimizer, pick_device, set_seed

HERE = Path(__file__).parent
MiB = 2**20


def estimate_mib(cfg: dict, batch_size: int, n_letters: int, n_params: int) -> dict:
    m = cfg["model"]
    D, H, N = m["d_model"], m["n_heads"], m["n_layers"]
    side = 64 // 2 ** m["n_pools"]
    V = side * side if m["visual_tokens"] == "grid" else 1
    T = V + n_letters
    fixed = n_params * 4 * (1 + 1 + 2) / MiB                       # weights + grads + Adam m, v
    dec = N * (16 * T * D + 3 * H * T * T) * 4
    cnn, res = 0, 64
    for i, c in enumerate(m["enc_channels"]):
        cnn += 3 * c * res * res * 4
        if i < m["n_pools"]:
            res //= 2
    return {"fixed_MiB": fixed, "activations_MiB": batch_size * (dec + cnn) / MiB,
            "total_MiB": fixed + batch_size * (dec + cnn) / MiB}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512])
    a = p.parse_args()
    device = pick_device(a.device)
    cfg = load_config(a.config); set_seed(0)
    tok = CharTokenizer(); ds = load_split(cfg["data_dir"], "train")
    (HERE / "hardware_memory.txt").write_text(hardware_info() + "\n")
    rows = []
    for bs in a.batch_sizes:
        batch = next(iter(make_loader(ds, tok, bs, True, seed=0)))
        n_letters = batch["text_in"].size(1)
        model = build_model(cfg).to(device).train()
        n_params = sum(p_.numel() for p_ in model.parameters())
        est = estimate_mib(cfg, bs, n_letters, n_params)
        measured = float("nan")
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            opt = make_optimizer(model, 1e-3, 0.01)
            loss = compute_loss(model, batch, device)
            opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize()
            measured = torch.cuda.max_memory_allocated() / MiB
            del opt, loss
        rows.append({"batch_size": bs, "n_letters": n_letters, **{k: round(v, 1) for k, v in est.items()},
                     "measured_peak_MiB": round(measured, 1)})
        print(rows[-1])
        del model
    with open(HERE / f"memory_{device.type}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if device.type != "cuda":
        print("NOTE: no CUDA device - only the estimate was computed; run on a GPU for the measured column.")


if __name__ == "__main__":
    main()
