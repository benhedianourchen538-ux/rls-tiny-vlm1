"""Multi-head self-attention written with plain tensor operations.

Pipeline (B = batch, T = sequence length, D = d_model, H = heads, dh = D / H):
    x (B,T,D) --qkv Linear--> (B,T,3D) --split--> q,k,v (B,T,D)
      --split heads--> (B,H,T,dh)
      scores = q @ k^T / sqrt(dh)             (B,H,T,T)
      scores = masked_fill(~mask, -inf)       mask: True = "may attend"
      weights = softmax(scores, dim=-1)       each row sums to 1
      y = weights @ v                         (B,H,T,dh)
      --merge heads--> (B,T,D) --out Linear--> (B,T,D)

Why divide by sqrt(dh)? q.k is a sum of dh products; with unit-variance entries its variance is dh,
so scores grow with dh and softmax saturates (near one-hot -> tiny gradients). Dividing by sqrt(dh)
brings the variance back to ~1.

Memory grows like B*H*T^2 because the full (T,T) score matrix is materialised (and kept for backward).
"""
import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)   # Q, K, V projections fused in one matmul
        self.out = nn.Linear(d_model, d_model)       # output projection (mixes the heads)
        self.attn_drop = nn.Dropout(dropout)
        self.store_attn = False                      # set True to keep the weights (interpretability)
        self.last_attn = None                        # (B, H, T, T) when store_attn is True

    # ------------------------------------------------------------------ helpers
    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)    # (B, H, T, dh)

    def merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, H, T, dh = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, H * dh)           # (B, T, D)

    def _attend(self, q, k, v, mask):
        """Scaled dot-product attention core. q,k,v: (B,H,T,dh). Returns (B,H,T,dh).

        Overridden in benchmarks to compare with F.scaled_dot_product_attention.
        """
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)           # (B, H, T, T)
        if mask is not None:
            # every row of our masks contains at least the diagonal, so no row is fully -inf (-> NaN)
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)                              # (B, H, T, T)
        if self.store_attn:
            self.last_attn = weights.detach()
        return self.attn_drop(weights) @ v                                   # (B, H, T, dh)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D). mask: bool, broadcastable to (B, H, T, T), True = position may be attended."""
        q, k, v = self.qkv(x).chunk(3, dim=-1)                               # 3 x (B, T, D)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        y = self._attend(q, k, v, mask)                                      # (B, H, T, dh)
        return self.out(self.merge_heads(y))                                 # (B, T, D)
