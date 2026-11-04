"""Train + evaluate one config over several seeds and aggregate mean +- std (needed for every accuracy claim).

    python run_seeds.py --config configs/baseline.yaml --name baseline --seeds 0 1 2 --out experiments/main
    python run_seeds.py --config configs/blind.yaml    --name blind    --seeds 0 1 2 --out experiments/E1_blind

For each seed: train -> runs/<name>_s<seed>/, then evaluate best.pt on `test` and `test_heldout`.
Writes <out>/<name>_results.json (per-seed numbers + mean/std).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import load_split, make_loader
from src.evaluate import evaluate
from src.tokenizer import CharTokenizer
from src.train import build_model, load_config, pick_device, train


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and k != "n":
            yield prefix + k, float(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--splits", nargs="+", default=["test", "test_heldout"])
    a = p.parse_args()
    device = pick_device(a.device)
    tok = CharTokenizer()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for seed in a.seeds:
        cfg = load_config(a.config, a.set); cfg["seed"] = seed
        run_dir = f"runs/{a.name}_s{seed}"
        resume = Path(run_dir, "last.pt").exists()                 # re-running the command resumes (Colab)
        train(cfg, run_dir, device, resume=resume)
        ck = torch.load(f"{run_dir}/best.pt", map_location="cpu", weights_only=False)
        model = build_model(ck["config"]).to(device); model.load_state_dict(ck["model"])
        per_seed[seed] = {}
        for split in a.splits:
            loader = make_loader(load_split(cfg["data_dir"], split), tok, 256, shuffle=False)
            per_seed[seed][split] = evaluate(model, loader, tok, device, cfg.get("image_mode", "normal"))
            print(seed, split, json.dumps(per_seed[seed][split]))
    summary = {}
    for split in a.splits:
        keys = dict(flatten(per_seed[a.seeds[0]][split])).keys()
        summary[split] = {k: {"mean": float(np.mean([dict(flatten(per_seed[s][split]))[k] for s in a.seeds])),
                              "std": float(np.std([dict(flatten(per_seed[s][split]))[k] for s in a.seeds]))}
                          for k in keys}
    json.dump({"config": a.config, "seeds": a.seeds, "overrides": a.set, "per_seed": per_seed, "summary": summary},
              open(out / f"{a.name}_results.json", "w"), indent=1)
    print(f"\n{a.name}: test exact match = {summary['test']['exact_match']['mean']:.4f} "
          f"+- {summary['test']['exact_match']['std']:.4f} (n={len(a.seeds)} seeds, population std)")


if __name__ == "__main__":
    main()
