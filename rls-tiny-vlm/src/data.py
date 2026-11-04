"""Dataset, collate function and DataLoader for ShapeScenes.

`generate_data.py` writes one file per split: {"images": uint8 (N, 3, 64, 64), "words": list[str]}.
Images stay uint8 in memory and during the CPU->GPU copy (4x smaller than float32); the CNN encoder
converts them to float on the device.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import CharTokenizer


class ShapeScenesDataset(Dataset):
    def __init__(self, images: torch.Tensor, words: list[str]):
        assert len(images) == len(words)
        self.images = images  # (N, 3, 64, 64) uint8
        self.words = words

    def __len__(self):
        return len(self.words)

    def __getitem__(self, i):
        return self.images[i], self.words[i]


def load_split(data_dir: str, split: str) -> ShapeScenesDataset:
    blob = torch.load(Path(data_dir) / f"{split}.pt", map_location="cpu", weights_only=False)
    return ShapeScenesDataset(blob["images"], blob["words"])


class Collator:
    """Stack images; turn words of different lengths into padded (text_in, targets)."""

    def __init__(self, tokenizer: CharTokenizer):
        self.tok = tokenizer

    def __call__(self, items):
        images = torch.stack([it[0] for it in items])      # (B, 3, 64, 64) uint8
        words = [it[1] for it in items]
        text_in, targets = self.tok.batch(words)            # (B, L), (B, L+1)
        return {"images": images, "text_in": text_in, "targets": targets, "words": words}


def make_loader(dataset, tokenizer, batch_size, shuffle, num_workers=0, pin_memory=False, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)  # reproducible shuffling
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, collate_fn=Collator(tokenizer), generator=g,
                      persistent_workers=num_workers > 0)
