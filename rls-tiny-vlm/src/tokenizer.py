"""Character-level tokenizer: 26 letters (a-z) + <eos> = 27 tokens.

Design notes
------------
* No start token: the output at the LAST visual token predicts the first letter.
* No pad token in the vocabulary: inputs are padded with an arbitrary id (``pad_id``)
  and the matching targets are set to ``IGNORE_INDEX`` so the loss skips them.
  Because padding is always on the RIGHT and the decoder is causal, real positions
  never attend to padded ones, so no padding mask is required.
"""
import string

import torch

IGNORE_INDEX = -100  # value ignored by nn.CrossEntropyLoss(ignore_index=...)


class CharTokenizer:
    letters = string.ascii_lowercase

    def __init__(self, max_len: int = 45):
        self.stoi = {c: i for i, c in enumerate(self.letters)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.eos_id = len(self.letters)          # 26
        self.pad_id = 0                          # arbitrary; never contributes to the loss
        self.vocab_size = self.eos_id + 1        # 27
        self.max_len = max_len

    # ------------------------------------------------------------------ single word
    def encode(self, word: str, add_eos: bool = False) -> list[int]:
        try:
            ids = [self.stoi[c] for c in word]
        except KeyError as e:
            raise KeyError(f"character {e} is not in a-z (words must be lowercase, no spaces)") from None
        return ids + [self.eos_id] if add_eos else ids

    def decode(self, ids) -> str:
        """Ids -> string. Stops at the first <eos>; ignores out-of-range ids."""
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i in self.itos:
                out.append(self.itos[i])
        return "".join(out)

    # ------------------------------------------------------------------ batch
    def batch(self, words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Words of different lengths -> (text_in, targets).

        text_in : (B, L)    letters w1..wn (right-padded with pad_id), L = max word length
        targets : (B, L+1)  w1..wn, <eos>, then IGNORE_INDEX

        The model produces L+1 logits per sample (from the last visual token up to the
        last letter), so logits[b, k] is trained to predict targets[b, k].
        """
        B = len(words)
        encoded = [self.encode(w) for w in words]
        L = max(len(e) for e in encoded)
        text_in = torch.full((B, L), self.pad_id, dtype=torch.long)
        targets = torch.full((B, L + 1), IGNORE_INDEX, dtype=torch.long)
        for b, ids in enumerate(encoded):
            n = len(ids)
            text_in[b, :n] = torch.tensor(ids)
            targets[b, :n] = torch.tensor(ids)
            targets[b, n] = self.eos_id
        return text_in, targets
