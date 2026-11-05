"""Inference helpers: load a checkpoint, turn ANY picture into the model's input, predict a word.

    python -m src.predict --ckpt runs/b0/best.pt --image my_scene.png
"""
import argparse

import numpy as np
import torch
from PIL import Image

from src.generate import greedy_generate
from src.parsing import parse_word
from src.tokenizer import CharTokenizer
from src.train import build_model, pick_device

NAMES = {"leftof": "left of", "rightof": "right of"}


def load_model(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Any PIL image -> (1, 3, 64, 64) uint8, the format of the training data (RGB, 64x64)."""
    img = img.convert("RGB").resize((64, 64), Image.LANCZOS)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).contiguous()


def describe(word: str) -> str:
    """'largeredcircleleftofsmallbluesquare' -> 'large red circle left of small blue square'."""
    parts = parse_word(word).parts
    return " ".join(NAMES.get(p, p) for p in parts)


@torch.no_grad()
def predict(model, img: Image.Image, device: torch.device) -> str:
    return greedy_generate(model, image_to_tensor(img).to(device), CharTokenizer())[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    device = pick_device(a.device)
    word = predict(load_model(a.ckpt, device), Image.open(a.image), device)
    print(word, "->", describe(word))


if __name__ == "__main__":
    main()
