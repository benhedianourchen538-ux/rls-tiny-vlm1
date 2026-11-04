"""Greedy decoding: one letter per step until <eos> (or 45 letters)."""
import torch

from src.tokenizer import CharTokenizer


@torch.no_grad()
def greedy_generate(model, images: torch.Tensor, tokenizer: CharTokenizer, max_len: int = 45) -> list[str]:
    """images (B,3,64,64) -> list of B words. The image is encoded once; the letters are re-fed each step
    (sequences are < 110 tokens, so no KV-cache is needed)."""
    was_training = model.training
    model.eval()
    visual = model.encode_image(images)                                       # (B, V, D)
    B, dev = visual.size(0), visual.device
    text = torch.empty(B, 0, dtype=torch.long, device=dev)                    # (B, t) letters so far
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(max_len):
        logits = model.decode(visual, text)                                   # (B, t+1, 27)
        nxt = logits[:, -1].argmax(dim=-1)                                    # (B,)
        nxt = torch.where(finished, torch.full_like(nxt, tokenizer.pad_id), nxt)
        finished |= nxt == tokenizer.eos_id
        text = torch.cat([text, nxt[:, None]], dim=1)
        if finished.all():
            break
    model.train(was_training)
    return [tokenizer.decode(row.tolist()) for row in text]                   # decode() stops at <eos>
