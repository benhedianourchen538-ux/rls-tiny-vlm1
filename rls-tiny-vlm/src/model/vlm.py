"""Tiny vision-language model: CNN encoder -> adapter -> Transformer decoder -> 27 logits.

Shapes (B batch, V visual tokens, L letters in the batch, D d_model):
    images     (B, 3, 64, 64)  uint8
    feature map(B, C, 8, 8)               CNNEncoder
    visual     (B, V=64, D)               Adapter: flatten H'W' + Linear(C -> D)
    letters    (B, L)  ids  -> (B, L, D)  token embedding
    sequence   (B, V+L, D)                [visual | letters]
    decoder    (B, V+L, D)
    logits     (B, L+1, 27)               head applied from the LAST visual token onward:
                                          position V-1 predicts letter 1, position V-1+k predicts letter k+1,
                                          position V-1+n predicts <eos>.
"""
from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from src.model.decoder import TransformerDecoder
from src.model.encoder import CNNEncoder


@dataclass
class VLMConfig:
    img_size: int = 64
    enc_channels: tuple = (32, 64, 128, 128)
    n_pools: int = 3
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    visual_tokens: str = "grid"      # "grid": H'*W' tokens | "global": 1 pooled token (experiment E3)
    visual_causal: bool = False      # causal attention among visual tokens (experiment E3)
    vocab_size: int = 27
    max_text_len: int = 45

    @classmethod
    def from_dict(cls, d: dict) -> "VLMConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown model config keys: {sorted(unknown)}")
        d = dict(d)
        if "enc_channels" in d:
            d["enc_channels"] = tuple(d["enc_channels"])
        return cls(**d)


class Adapter(nn.Module):
    """Feature map -> sequence of d_model vectors (the 'visual tokens')."""

    def __init__(self, in_ch: int, d_model: int, mode: str = "grid"):
        super().__init__()
        assert mode in ("grid", "global")
        self.mode = mode
        self.proj = nn.Linear(in_ch, d_model)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        if self.mode == "global":
            fmap = fmap.mean(dim=(2, 3), keepdim=True)         # (B, C, 1, 1)
        tokens = fmap.flatten(2).transpose(1, 2)               # (B, C, H, W) -> (B, H*W, C)
        return self.proj(tokens)                               # (B, V, D)


def _init(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)


class VLM(nn.Module):
    def __init__(self, cfg: VLMConfig | None = None):
        super().__init__()
        cfg = cfg or VLMConfig()          # default arguments = baseline architecture
        self.cfg = cfg
        self.encoder = CNNEncoder(cfg.enc_channels, cfg.n_pools)
        side = self.encoder.out_size(cfg.img_size)
        self.n_visual = side * side if cfg.visual_tokens == "grid" else 1
        self.adapter = Adapter(self.encoder.out_channels, cfg.d_model, cfg.visual_tokens)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.decoder = TransformerDecoder(cfg.d_model, cfg.n_heads, cfg.n_layers,
                                          max_len=self.n_visual + cfg.max_text_len,
                                          dropout=cfg.dropout, visual_causal=cfg.visual_causal)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        for m in (self.adapter, self.tok_emb, self.decoder, self.head):
            _init(m)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """(B,3,64,64) -> visual tokens (B, V, D). Computed once and reused during generation."""
        return self.adapter(self.encoder(images))

    def decode(self, visual: torch.Tensor, text_in: torch.Tensor) -> torch.Tensor:
        """visual (B,V,D), text_in (B,L) -> logits (B, L+1, vocab)."""
        txt = self.tok_emb(text_in)                            # (B, L, D)
        x = torch.cat([visual, txt], dim=1)                    # (B, V+L, D)
        h = self.decoder(x, n_visual=visual.size(1))           # (B, V+L, D)
        return self.head(h[:, visual.size(1) - 1:])            # (B, L+1, vocab)

    def forward(self, images: torch.Tensor, text_in: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode_image(images), text_in)
