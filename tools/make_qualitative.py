"""Qualitative grid: rectified crops with the ground truth and both models' text.

The crops are rectified with the **ground-truth** quad so the comparison isolates
the recogniser: any difference is the province glyph, not detection quality.

    python tools/make_qualitative.py --old runs/smoke_fix/checkpoints/best.pth \
                                     --new runs/plate_bal/checkpoints/best.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lpr.ccpd import CCPDDataset  # noqa: E402
from lpr.charset import ctc_greedy_decode  # noqa: E402
from lpr.geometry import warp_quads  # noqa: E402
from lpr.sampling import province_of  # noqa: E402
from predict import load_model  # noqa: E402

plt.rcParams.update({"font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def pick_samples(ds: CCPDDataset, n: int, prefer: list[str] | None = None) -> list[int]:
    """One index per distinct non-皖 province, plus a couple of 皖 for contrast."""
    seen: dict[str, int] = {}
    for i, f in enumerate(ds.files):
        p = province_of(f)
        if p and p not in seen:
            seen[p] = i
    want = prefer or [p for p in seen if p != "皖"]
    idx = [seen[p] for p in want if p in seen][:n]
    if "皖" in seen:
        idx.insert(min(2, len(idx)), seen["皖"])
    return idx[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="runs/smoke_fix/checkpoints/best.pth")
    ap.add_argument("--new", default="runs/plate_bal/checkpoints/best.pth")
    ap.add_argument("--split", default="test_prov")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="runs/report_figs/fig8_qualitative.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    models = {}
    for tag, path in (("old", args.old), ("new", args.new)):
        p = PROJECT_ROOT / path
        if p.is_file():
            models[tag] = load_model(p, device)
        else:
            print(f"[skip] {tag}: missing {p}")
    if "new" not in models:
        print("[FATAL] the new checkpoint is required")
        return 2

    ref = models["new"]
    ds = CCPDDataset(PROJECT_ROOT / "data" / "CCPD2019",
                     PROJECT_ROOT / "data" / "CCPD2019" / "splits" / f"{args.split}.txt",
                     input_hw=(ref.input_h, ref.input_w), train=False)
    idx = pick_samples(ds, args.n)

    cols = min(4, len(idx))
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.35 * cols, 2.5 * rows))
    axes = np.atleast_2d(axes)

    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        item = ds[i]
        image = item["image"][None]
        gt_px = (item["corners"][None] * torch.tensor([ref.input_w, ref.input_h]))[:, :, :]
        with torch.no_grad():
            crop = warp_quads(image, gt_px, ref.crop_hw)[0]
            texts = {}
            for tag, m in models.items():
                out = m(image.to(device))
                texts[tag] = ctc_greedy_decode(out["logits"])[0]
        img = crop.permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5) if img.min() < 0 else img
        ax.imshow(np.clip(img, 0, 1))
        gt = item["text"]
        old_t = texts.get("old", "—")
        new_t = texts.get("new", "—")
        ok_old = old_t == gt
        ok_new = new_t == gt
        ax.set_title(f"GT  {gt}", fontsize=10, color="#1b1f24", pad=4)
        ax.set_xlabel(f"旧 {old_t}  {'✓' if ok_old else '✗'}\n新 {new_t}  {'✓' if ok_new else '✗'}",
                      fontsize=9.5, color="#1b1f24", labelpad=3)
        print(f"  {Path(item['path']).name[:24]:26s} GT {gt:9s} | 旧 {old_t:9s}"
              f" {'✓' if ok_old else '✗'} | 新 {new_t:9s} {'✓' if ok_new else '✗'}")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#0f8a5f" if ok_new else "#c0392b")
            s.set_linewidth(2.0)
    for k in range(len(idx), rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.suptitle("用真值角点校正后的车牌：只比较识别分支", fontsize=12, y=0.99)
    fig.tight_layout()
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[fig] {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
