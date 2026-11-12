"""Province-balanced sampling for the CCPD long tail.

The problem
-----------
CCPD2019 is not merely "imbalanced", it is *pathologically* imbalanced on the
character that matters most: the province glyph in position 1.

===================  =========  ==========
province              images     share
===================  =========  ==========
皖 (Anhui)            326 570     92.8 %
苏                     7 697      2.2 %
浙                     3 565      1.0 %
...                         ...        ...
宁 (Ningxia)               8      0.002 %
藏 (Tibet)                 1      0.0003 %
===================  =========  ==========

Train on that and the empirically optimal recogniser outputs 皖 for anything
ambiguous -- which is exactly the failure mode observed in production.

Why re-weighting, and not just "more data"
------------------------------------------
The empirical risk minimiser reproduces the *training* prior. With
``P_train(皖) = 0.93`` and, say, ``P_deploy(皖) ≈ 0.1``, the Bayes-optimal
decision barely moves: the 皖 logit is inflated by

    Δ = log(0.93 / 0.10) ≈ 2.2 nats   (≈ 9x likelihood ratio)

i.e. a genuinely non-皖 plate must look ``e^2.2 ≈ 9`` times more likely under its
own class before the argmax escapes 皖. Adding the other sub-directories helps a
little (they are 83-96 % 皖 themselves, so the prior barely changes), but it
cannot fix a factor of 9 in the logit space. The sampling distribution has to be
changed.

The sampler
-----------
Draw sample ``i`` with probability

    q_i ∝ r_{f(i)} · N_{p(i)}^{−α}

where ``N_p`` is the number of training images of province ``p``, ``r_f`` an
optional per-folder repeat factor and ``α ∈ [0, 1]`` the re-balancing strength.
Because ``q_i · #{i : p(i)=p} ∝ N_p^{1−α}``, the *effective* province
distribution becomes

    P_eff(p) ∝ N_p^{1 − α}

so ``α = 0`` reproduces the natural prior, ``α = 1`` is fully uniform and
``α = 0.5`` is the square-root compromise. ``α = 0.5`` is the value recommended
by the "effective number of samples" analysis of Cui et al. (CVPR 2019) for
tails of this size (their β ≈ 0.9999 gives exactly a ``N^{−1/2}`` weight), and it
keeps 皖 at a still-useful ~35 % of the batch instead of starving it.

Two guards
----------
``weight_cap``      single-image weight clipped to ``cap × median(w)``. Without
                    it 藏 (one image in the entire dataset) would be drawn
                    thousands of times per epoch and simply memorised.
``province_floor``  provinces with fewer than ``floor`` images are treated as if
                    they had ``floor``, which bounds the amplification the same
                    way and keeps the tail from dominating the batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .charset import PROVINCES


def province_of(path: str) -> str:
    """Province character of a CCPD file, from its filename annotation."""
    parts = Path(path).stem.split("-")
    if len(parts) < 5:
        return "?"
    try:
        idx = int(parts[4].split("_")[0])
    except (ValueError, IndexError):
        return "?"
    return PROVINCES[idx] if 0 <= idx < len(PROVINCES) else "?"


def folder_of(path: str) -> str:
    """Top-level sub-directory a CCPD path lives in."""
    parts = Path(path).parts
    for p in parts[:-1]:
        if p.startswith("ccpd_"):
            return p
    return parts[-2] if len(parts) >= 2 else "?"


@dataclass
class SamplingPlan:
    """Per-file sampling weights plus the diagnostics the report needs."""

    weights: np.ndarray
    provinces: List[str]
    folders: List[str]
    alpha: float
    natural: Dict[str, float] = field(default_factory=dict)
    effective: Dict[str, float] = field(default_factory=dict)
    folder_effective: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        top = sorted(self.effective.items(), key=lambda kv: -kv[1])[:5]
        tail = sorted(self.effective.items(), key=lambda kv: kv[1])[:3]
        eff = " ".join(f"{k}:{v:.1%}" for k, v in top)
        low = " ".join(f"{k}:{v:.2%}" for k, v in tail)
        ratio = self.effective.get("皖", 0.0) / max(self.natural.get("皖", 1e-9), 1e-9)
        return (f"[sampler] alpha={self.alpha} 皖 natural={self.natural.get('皖', 0):.1%} "
                f"-> effective={self.effective.get('皖', 0):.1%} (x{ratio:.2f})\n"
                f"          top: {eff}\n          tail: {low}")


def build_sampling_plan(
    files: Sequence[str],
    alpha: float = 0.5,
    folder_repeat: Dict[str, float] | None = None,
    province_floor: int = 32,
    weight_cap: float = 0.0,
) -> SamplingPlan:
    """Inverse-frequency weights ``r_f · max(N_p, floor)^{−α}``, optionally capped.

    Args:
        files: training file paths (relative or absolute, CCPD naming)
        alpha: re-balancing strength in ``[0, 1]``; 0 = natural, 1 = uniform
        folder_repeat: ``{sub_dir: multiplier}``, e.g. ``{"ccpd_tilt": 2.0}``
        province_floor: provinces below this count are weighted as if they had it
        weight_cap: optional cap on ``w_i / median(w)`` (0 disables). Note that the
            median is dominated by 皖, so any modest cap silently flattens the
            whole distribution -- prefer ``province_floor``.
    """
    provinces = [province_of(f) for f in files]
    folders = [folder_of(f) for f in files]
    folder_repeat = folder_repeat or {}

    counts: Dict[str, int] = {}
    for p in provinces:
        counts[p] = counts.get(p, 0) + 1

    w = np.empty(len(files), dtype=np.float64)
    for i, (p, f) in enumerate(zip(provinces, folders)):
        n_eff = max(counts[p], province_floor)
        w[i] = folder_repeat.get(f, 1.0) * n_eff ** (-alpha)

    med = float(np.median(w))
    if weight_cap and med > 0:
        w = np.minimum(w, weight_cap * med)
    w /= w.sum()

    # diagnostics: expected share of each province under this multinomial
    eff: Dict[str, float] = {}
    for i, p in enumerate(provinces):
        eff[p] = eff.get(p, 0.0) + float(w[i])
    folder_eff: Dict[str, float] = {}
    for i, f in enumerate(folders):
        folder_eff[f] = folder_eff.get(f, 0.0) + float(w[i])
    total = sum(counts.values())
    natural = {p: c / total for p, c in counts.items()}

    return SamplingPlan(weights=w, provinces=provinces, folders=folders, alpha=alpha,
                        natural=natural, effective=eff, folder_effective=folder_eff)
