"""Transformer decoder over the sequence [visual tokens | letter tokens].

Attention mask ("prefix-LM"):
    * letter tokens are causal: a letter sees all visual tokens and the letters before it;
    * visual tokens see all visual tokens (bidirectional) and never see letters.
      With ``visual_causal=True`` they become causal among themselves (experiment idea E3).
mask[i, j] = True  means  "position i may attend to position j".

Block (pre-LayerNorm):   x = x + Attn(LN(x));   x = x + MLP(LN(x))
"""
import torch
import torch.nn as nn

from src.model.attention import MultiHeadSelfAttention


def build_prefix_mask(n_visual: int, n_text: int, visual_causal: bool = False, device=None) -> torch.Tensor:
    T = n_visual + n_text
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))     # causal everywhere
    if not visual_causal:
        mask[:n_visual, :n_visual] = True                                     # visual block fully visible
    return mask                                                               # (T, T)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):                         # x: (B, T, D)
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, max_len, dropout=0.1, visual_causal=False):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)    # learned positions for visual AND letter slots
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.visual_causal = visual_causal

    def forward(self, x: torch.Tensor, n_visual: int) -> torch.Tensor:
        B, T, D = x.shape                                # x = [visual | letters] embeddings
        x = self.drop(x + self.pos_emb(torch.arange(T, device=x.device)))    # (B, T, D)
        mask = build_prefix_mask(n_visual, T - n_visual, self.visual_causal, x.device)  # (T, T)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.ln_f(x)                              # (B, T, D)
