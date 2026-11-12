"""Character set for CCPD Chinese license plates.

CCPD encodes the plate text in the image filename as dash-separated integers.
The table is **not** "A-Z + 0-9": Chinese plates never use the letters ``I`` and
``O``, so CCPD uses 24 letters followed by 10 digits (34 symbols), giving
31 provinces + 34 symbols = 65 classes, plus one CTC blank = 66 logits.

Verified against the images: ``皖A·PV930`` carries labels ``[0, 0, 13, 19, 33, 27, 24]``
-> A=0, P=13, V=19, 9=33, 3=27, 0=24.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

PROVINCES: List[str] = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
    "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
    "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
]

# 24 letters (no I, no O) then 10 digits -- CCPD's actual label ordering.
ALPHABETS: List[str] = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + list("0123456789")
CHARSET: List[str] = PROVINCES + ALPHABETS

BLANK_INDEX: int = len(CHARSET)      # 65
NUM_CLASSES: int = len(CHARSET) + 1  # 66 (0..64 chars, 65 blank)

assert len(PROVINCES) == 31 and len(ALPHABETS) == 34 and NUM_CLASSES == 66


def decode_label_indices(indices: Sequence[int]) -> str:
    """``[0, 0, 13, 19, 33, 27, 24]`` -> ``"皖APV930"``."""
    out = []
    for k, i in enumerate(indices):
        i = int(i)
        if k == 0:
            out.append(PROVINCES[i] if 0 <= i < len(PROVINCES) else "?")
        else:
            out.append(ALPHABETS[i] if 0 <= i < len(ALPHABETS) else "?")
    return "".join(out)


def encode_text(text: str) -> List[int]:
    """``"皖APV930"`` -> ``[0, 0, 13, 19, 33, 27, 24]``."""
    ids: List[int] = []
    for k, ch in enumerate(text):
        if k == 0 and ch in PROVINCES:
            ids.append(PROVINCES.index(ch))
        elif ch in ALPHABETS:
            ids.append(ALPHABETS.index(ch))
        else:
            raise ValueError(f"unknown character {ch!r} at position {k} in {text!r}")
    return ids


def ctc_greedy_decode(logits, input_lengths: Iterable[int] | None = None) -> List[str]:
    """Greedy CTC decode of ``(B, T, C)`` logits."""
    import torch  # local import keeps the module importable without torch

    if logits.dim() != 3:
        raise ValueError("logits must be 3-D")
    preds = logits.argmax(dim=-1)  # (B, T)
    results: List[str] = []
    for b in range(preds.shape[0]):
        collapsed: List[int] = []
        prev = -1
        for tok in preds[b].tolist():
            if tok != prev and tok != BLANK_INDEX:
                collapsed.append(int(tok))
            prev = tok
        results.append(decode_label_indices(collapsed))
    return results
