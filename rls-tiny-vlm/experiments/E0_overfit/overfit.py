"""E0 - overfit ONE batch (a bug check, not a result).

If the pipeline is correct, a 1M-parameter model must drive the loss on a single fixed batch to ~0 and
reproduce all of its words. If it cannot, the bug is in data / loss masking / attention mask / optimizer,
not in "model capacity". Dropout is switched off so the batch can be memorised.

    python -m experiments.E0_overfit.overfit --device cpu
"""
import argparse
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import load_split, make_loader  # noqa: E402
from src.generate import greedy_generate  # noqa: E402
from src.tokenizer import CharTokenizer  # noqa: E402
from src.train import build_model, compute_loss, load_config, make_optimizer, pick_device, set_seed  # noqa: E402

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()
    device = pick_device(a.device)
    cfg = load_config(a.config, ["model.dropout=0.0"])
    set_seed(cfg["seed"])
    tok = CharTokenizer()
    batch = next(iter(make_loader(load_split(cfg["data_dir"], "train"), tok, a.batch_size, shuffle=False)))
    model = build_model(cfg).to(device).train()
    opt = make_optimizer(model, a.lr, 0.0)
    losses = []
    for step in range(a.steps):
        loss = compute_loss(model, batch, device)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == a.steps - 1:
            print(f"step {step:4d}  loss {losses[-1]:.4f}")
    preds = greedy_generate(model, batch["images"].to(device), tok)   # eval mode inside
    exact = sum(p_ == w for p_, w in zip(preds, batch["words"])) / len(preds)
    print(f"final loss {losses[-1]:.5f} | exact match on the memorised batch: {exact:.2%}")
    for p_, w in list(zip(preds, batch["words"]))[:3]:
        print(f"  ref {w}\n  gen {p_}")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(losses); ax.set(yscale="log", xlabel="step", ylabel="cross-entropy (log scale)",
                            title=f"E0: overfitting one batch of {a.batch_size}")
    ax.grid(alpha=0.3, which="both"); fig.tight_layout(); fig.savefig(HERE / "overfit_loss.png", dpi=150)
    json.dump({"device": device.type, "steps": a.steps, "batch_size": a.batch_size, "final_loss": losses[-1],
               "exact_match_on_batch": exact, "loss_every_20": losses[::20]}, open(HERE / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
