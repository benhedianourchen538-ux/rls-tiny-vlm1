"""Training loop: forward -> masked cross-entropy -> backward -> optimizer step, with validation,
checkpoints (resumable: Colab sessions die) and loss curves.

    python -m src.train --config configs/baseline.yaml --seed 0 --out runs/baseline_s0
    python -m src.train --config configs/baseline.yaml --out runs/baseline_s0 --resume
    python -m src.train --config configs/baseline.yaml --set train.epochs=1 --set train.batch_size=32
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.data import load_split, make_loader
from src.evaluate import apply_image_mode, evaluate
from src.model.vlm import VLM, VLMConfig
from src.tokenizer import IGNORE_INDEX, CharTokenizer


# ----------------------------------------------------------------------------- helpers
def pick_device(flag: str = "auto") -> torch.device:
    """One flag switches CPU/GPU: --device cpu | cuda | auto."""
    if flag == "auto":
        flag = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(flag)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> VLM:
    return VLM(VLMConfig.from_dict(cfg["model"]))


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:                       # --set train.epochs=3
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def compute_loss(model, batch, device, image_mode="normal", autocast=None):
    """Cross-entropy over all letter positions; padded positions have target IGNORE_INDEX."""
    images = apply_image_mode(batch["images"].to(device, non_blocking=True), image_mode)   # (B,3,64,64)
    text_in = batch["text_in"].to(device, non_blocking=True)                                # (B, L)
    targets = batch["targets"].to(device, non_blocking=True)                                # (B, L+1)
    with autocast or torch.autocast(device.type, enabled=False):
        logits = model(images, text_in)                                                     # (B, L+1, 27)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=IGNORE_INDEX)


@torch.no_grad()
def validation_loss(model, loader, device, image_mode):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        total += compute_loss(model, batch, device, image_mode).item() * len(batch["words"])
        n += len(batch["words"])
    return total / n


def make_scheduler(opt, warmup: int, total: int):
    def lr_lambda(step):                                # linear warm-up, then cosine decay to 10%
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def make_optimizer(model, lr, weight_decay):
    decay = [p for p in model.parameters() if p.ndim >= 2]        # weights get decay ...
    no_decay = [p for p in model.parameters() if p.ndim < 2]      # ... biases / LayerNorm / BatchNorm do not
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def plot_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy", title="Loss"); ax[0].legend()
    ax[1].plot(ep, [h["val_exact_match"] for h in history])
    ax[1].set(xlabel="epoch", ylabel="exact match (val subset)", title="Validation exact match", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- main loop
def train(cfg: dict, out_dir: str, device: torch.device, resume: bool = False) -> list[dict]:
    t = cfg["train"]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    tok = CharTokenizer()
    image_mode = cfg.get("image_mode", "normal")
    pin = device.type == "cuda"
    train_loader = make_loader(load_split(cfg["data_dir"], "train"), tok, t["batch_size"], True,
                               t.get("num_workers", 0), pin, seed=cfg["seed"])
    val_loader = make_loader(load_split(cfg["data_dir"], "val"), tok, 256, False)

    model = build_model(cfg).to(device)
    opt = make_optimizer(model, t["lr"], t["weight_decay"])
    total_steps = t["epochs"] * len(train_loader)
    sched = make_scheduler(opt, t["warmup_steps"], total_steps)
    use_amp = bool(t.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=use_amp)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,} steps/epoch={len(train_loader)} image_mode={image_mode}")

    history, start_epoch, best = [], 0, float("inf")
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        history, start_epoch, best = ck["history"], ck["epoch"], ck["best"]
        print(f"resumed from epoch {start_epoch}")

    def save(name, epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "history": history, "best": best,
                    "config": cfg}, out / name)

    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        for batch in train_loader:
            loss = compute_loss(model, batch, device, image_mode, autocast)
            opt.zero_grad(set_to_none=True)             # clear gradients of the previous step
            scaler.scale(loss).backward()               # autograd: fills p.grad for every parameter
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt)                            # AdamW update from p.grad
            scaler.update()
            sched.step()
            running += loss.item(); nb += 1
        val_loss = validation_loss(model, val_loader, device, image_mode)
        val_em = evaluate(model, val_loader, tok, device, image_mode, max_batches=t.get("val_gen_batches", 2))["exact_match"]
        rec = {"epoch": epoch + 1, "train_loss": running / nb, "val_loss": val_loss, "val_exact_match": val_em,
               "lr": sched.get_last_lr()[0], "epoch_seconds": time.time() - t0}
        history.append(rec)
        print(json.dumps(rec))
        if val_loss < best:
            best = val_loss
            save("best.pt", epoch + 1)
        save("last.pt", epoch + 1)
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=1)
    plot_curves(history, out / "curves.png")
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--set", action="append", default=[], help="override a config value, e.g. train.epochs=3")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    if a.seed is not None:
        cfg["seed"] = a.seed
    train(cfg, a.out, pick_device(a.device), a.resume)


if __name__ == "__main__":
    main()
