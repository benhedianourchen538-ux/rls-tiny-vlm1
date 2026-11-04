"""Evaluation: exact match, per-attribute accuracy, letter accuracy.

Three kinds of numbers, from most to least honest:
  * exact match      : generated word == reference word (greedy decoding).
  * attribute acc.   : the generated word is parsed back into size/color/shape/relation
                       (src/parsing.py survives misspellings) and compared slot by slot.
  * letter accuracy  : teacher-forced next-letter accuracy. It is INFLATED: once the model has
                       produced "ci", the letters "rcle" follow from spelling alone, no image needed.
                       We therefore also report it split into first-letter-of-a-part vs inside-a-part.
Run as a script:  python -m src.evaluate --ckpt runs/baseline/best.pt --split test
"""
import argparse
import json
from collections import defaultdict

import torch

from src.data import load_split, make_loader
from src.generate import greedy_generate
from src.parsing import attribute_correct, category, part_starts, parse_word
from src.tokenizer import IGNORE_INDEX, CharTokenizer


def apply_image_mode(images: torch.Tensor, mode: str) -> torch.Tensor:
    """Blind-baseline hook (E1). 'zeros': no information. 'shuffle': real images, wrong pairing."""
    if mode == "zeros":
        return torch.zeros_like(images)
    if mode == "shuffle":
        return images[torch.randperm(len(images), device=images.device)]
    return images


@torch.no_grad()
def evaluate(model, loader, tok: CharTokenizer, device, image_mode: str = "normal", max_batches: int | None = None) -> dict:
    model.eval()
    n = exact = 0
    parse_cost = 0
    slot_hits, slot_tot = defaultdict(int), defaultdict(int)       # keyed by category: size/color/shape/relation
    obj_hits, obj_tot = defaultdict(int), defaultdict(int)         # keyed by n_objects (1 or 2)
    letter_hits = letter_tot = 0
    first_hits = first_tot = inner_hits = inner_tot = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        images = apply_image_mode(batch["images"].to(device), image_mode)
        preds = greedy_generate(model, images, tok)
        # teacher-forced letter accuracy
        logits = model(images, batch["text_in"].to(device))             # (B, L+1, 27)
        targets = batch["targets"].to(device)                           # (B, L+1)
        correct = (logits.argmax(-1) == targets) & (targets != IGNORE_INDEX)
        valid = targets != IGNORE_INDEX
        letter_hits += correct.sum().item()
        letter_tot += valid.sum().item()
        for b, (pred, ref) in enumerate(zip(preds, batch["words"])):
            n += 1
            exact += pred == ref
            parse_cost += parse_word(pred).cost
            n_obj = parse_word(ref).n_objects
            ok = attribute_correct(pred, ref)
            for slot, hit in ok.items():
                slot_hits[category(slot)] += hit
                slot_tot[category(slot)] += 1
            obj_hits[n_obj] += pred == ref
            obj_tot[n_obj] += 1
            starts = set(part_starts(ref))
            for k in range(len(ref) + 1):                               # letters + <eos>
                if k == len(ref):
                    continue
                hit = bool(correct[b, k])
                if k in starts:
                    first_hits += hit
                    first_tot += 1
                else:
                    inner_hits += hit
                    inner_tot += 1
    return {
        "n": n,
        "exact_match": exact / n,
        "attribute_acc": {c: slot_hits[c] / slot_tot[c] for c in slot_tot},
        "exact_match_by_n_objects": {str(k): obj_hits[k] / obj_tot[k] for k in obj_tot},
        "letter_acc_teacher_forced": letter_hits / letter_tot,
        "letter_acc_first_of_part": first_hits / max(first_tot, 1),
        "letter_acc_inside_part": inner_hits / max(inner_tot, 1),
        "mean_parse_cost": parse_cost / n,
    }


def main():
    from src.train import build_model, pick_device  # local import: train imports evaluate
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["val", "test", "test_heldout"])
    p.add_argument("--data_dir", default="data")
    p.add_argument("--device", default="auto")
    p.add_argument("--image_mode", default=None, help="override; default = the mode stored in the checkpoint")
    p.add_argument("--out", default=None, help="write the metrics to this json file")
    a = p.parse_args()
    device = pick_device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    tok = CharTokenizer()
    model = build_model(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    mode = a.image_mode or ck["config"].get("image_mode", "normal")
    loader = make_loader(load_split(a.data_dir, a.split), tok, 256, shuffle=False)
    res = evaluate(model, loader, tok, device, image_mode=mode)
    res.update(split=a.split, ckpt=a.ckpt, image_mode=mode)
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
