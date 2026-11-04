"""Split a generated word back into (size, color, shape [, relation, size, color, shape]).

The parser must survive broken outputs such as ``largeredcirle``. Strategy:
  1. fast path: exact regex match (cost 0);
  2. otherwise: dynamic programming that cuts the string into the slots of the grammar
     (3 slots for one object, 7 for two) and picks, for each slot, the vocabulary word
     with the smallest edit distance. The segmentation with the lowest TOTAL cost wins.
"""
import re
from dataclasses import dataclass
from functools import lru_cache

SIZES = ("small", "large")
COLORS = ("red", "green", "blue", "yellow")
SHAPES = ("circle", "square", "triangle", "cross")
RELATIONS = ("leftof", "rightof", "above", "below")

VOCABS_1 = (SIZES, COLORS, SHAPES)
VOCABS_2 = VOCABS_1 + (RELATIONS,) + VOCABS_1
SLOTS_1 = ("size1", "color1", "shape1")
SLOTS_2 = SLOTS_1 + ("relation", "size2", "color2", "shape2")

_INF = 10**9


def _alt(words):
    return "(" + "|".join(words) + ")"


_RE_1 = re.compile("".join(_alt(v) for v in VOCABS_1))
_RE_2 = re.compile("".join(_alt(v) for v in VOCABS_2))


@lru_cache(maxsize=None)
def _lev(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _segment(word: str, vocabs) -> tuple[int, tuple[str, ...]]:
    """Best segmentation of ``word`` into len(vocabs) parts. Returns (cost, parts)."""
    n, k = len(word), len(vocabs)
    dp = [[_INF] * (n + 1) for _ in range(k + 1)]
    back = [[None] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0
    for s in range(1, k + 1):
        for i in range(n + 1):
            for w in vocabs[s - 1]:
                lw = len(w)
                # a segment is allowed to be up to 3 letters shorter/longer than the vocabulary word
                lo, hi = max(0, i - lw - 3), i - max(0, lw - 3)
                for j in range(lo, hi + 1):
                    if dp[s - 1][j] >= _INF:
                        continue
                    c = dp[s - 1][j] + _lev(word[j:i], w)
                    if c < dp[s][i]:
                        dp[s][i] = c
                        back[s][i] = (j, w)
    if dp[k][n] >= _INF:
        return _INF, ("",) * k
    parts, i = [], n
    for s in range(k, 0, -1):
        j, w = back[s][i]
        parts.append(w)
        i = j
    return dp[k][n], tuple(reversed(parts))


@dataclass(frozen=True)
class Parsed:
    parts: tuple[str, ...]   # normalized vocabulary words, 3 or 7 of them ("" if unparsable)
    cost: int                # total edit distance to the closest valid scene (0 = exact)

    @property
    def n_objects(self) -> int:
        return 1 if len(self.parts) == 3 else 2

    @property
    def slots(self) -> dict[str, str]:
        names = SLOTS_1 if self.n_objects == 1 else SLOTS_2
        return dict(zip(names, self.parts))


def parse_word(word: str) -> Parsed:
    m = _RE_1.fullmatch(word)
    if m:
        return Parsed(m.groups(), 0)
    m = _RE_2.fullmatch(word)
    if m:
        return Parsed(m.groups(), 0)
    c1, p1 = _segment(word, VOCABS_1)
    c2, p2 = _segment(word, VOCABS_2)
    return Parsed(p1, c1) if c1 <= c2 else Parsed(p2, c2)


def category(slot: str) -> str:
    """'color2' -> 'color'."""
    return slot.rstrip("12")


def part_starts(word: str) -> list[int]:
    """Index of the first letter of every part of a (reference) word."""
    starts, pos = [], 0
    for p in parse_word(word).parts:
        starts.append(pos)
        pos += len(p)
    return starts


def attribute_correct(pred_word: str, ref_word: str) -> dict[str, bool]:
    """For every slot of the REFERENCE scene: did the prediction get it right?"""
    ref = parse_word(ref_word).slots
    pred = parse_word(pred_word).slots
    return {slot: pred.get(slot) == value for slot, value in ref.items()}
