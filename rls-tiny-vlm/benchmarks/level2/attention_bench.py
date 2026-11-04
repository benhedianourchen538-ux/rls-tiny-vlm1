"""Level 2 - our attention vs F.scaled_dot_product_attention (SDPA): speed and peak memory vs sequence length.

Our version materialises the (B, H, T, T) score AND weight matrices (kept for backward): memory ~ B*H*T^2.
SDPA dispatches to a fused kernel (FlashAttention / memory-efficient) that never writes the T x T matrix to
global memory, so it is lighter and faster once T is large. Forward + backward, causal mask.
Memory is only measurable on CUDA (torch.cuda.max_memory_allocated).

    python -m benchmarks.level2.attention_bench --device cuda
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.common import hardware_info, time_block
from src.model.attention import MultiHeadSelfAttention
from src.train import pick_device

HERE = Path(__file__).parent


class SDPAAttention(MultiHeadSelfAttention):
    """Same weights and API as ours; only the inner computation is replaced (allowed in benchmarks only)."""

    def _attend(self, q, k, v, mask):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--seq_lens", type=int, nargs="+", default=[109, 256, 512, 1024, 2048])
    p.add_argument("--iters", type=int, default=10)
    a = p.parse_args()
    device = pick_device(a.device)
    (HERE / f"hardware_attn_{device.type}.txt").write_text(hardware_info() + "\n")
    rows = []
    for T in a.seq_lens:
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        for name, cls in [("ours", MultiHeadSelfAttention), ("sdpa", SDPAAttention)]:
            torch.manual_seed(0)
            attn = cls(a.d_model, a.n_heads).to(device).train()
            x = torch.randn(a.batch_size, T, a.d_model, device=device, requires_grad=True)

            def step():
                attn(x, mask).sum().backward()

            if device.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
            ms = time_block(step, device, warmup=3, iters=a.iters) * 1e3
            peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if device.type == "cuda" else float("nan")
            est = 4 * a.batch_size * a.n_heads * T * T * 2 / 2**20     # scores + weights, float32
            rows.append({"device": device.type, "impl": name, "T": T, "ms_fwd_bwd": ms,
                         "peak_extra_MiB": peak, "TxT_matrices_estimate_MiB": est})
            print(rows[-1])
    with open(HERE / f"attention_{device.type}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
