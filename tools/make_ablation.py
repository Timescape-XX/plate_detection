"""Equal-budget ablation figure: same data pool, same schedule, only α differs.

Both runs share ``train_mix``, ``folder_repeat=1.4``, batch 32, lr 1.2e-3, and the
same cosine schedule.  The only knob that changes is ``province_alpha``:

* ``runs/plate_bal``  α = 0.7  (inverse-frequency re-sampling)
* ``runs/ctrl_a0``    α = 0.0  (natural 92.7%-皖 prior)

So the gap between the two curves at the *same step* is the effect of the
sampling distribution alone, with training budget held fixed.

    python tools/make_ablation.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

C_INK = "#1b1f24"
C_MUTED = "#6b7280"
C_GRID = "#e5e7eb"
C_A07 = "#2f6fd0"
C_A00 = "#c0392b"
C_ACC = "#0f8a5f"

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": C_GRID,
    "axes.labelcolor": C_INK,
    "text.color": C_INK,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def read_val(csv_path: Path) -> dict:
    """Validation rows from train_log.csv, keyed by metric."""
    out = {"step": [], "acc": [], "macro": [], "wan_fp": []}
    if not csv_path.is_file():
        return out
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("val_plate_acc"):
                continue
            out["step"].append(int(row["step"]))
            out["acc"].append(float(row["val_plate_acc"]))
            out["macro"].append(float(row.get("val_macro_head") or 0))
            out["wan_fp"].append(float(row.get("val_wan_fp") or 0))
    return out


def fig_ablation(a07: dict, a00: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0))
    metrics = (("acc", "val_mix 整牌准确率", (0, 1.0)),
               ("macro", "非皖头部省份宏平均", (0, 1.0)),
               ("wan_fp", "非皖被判成「皖」的比例", (0, 1.0)))
    for ax, (key, title, ylim) in zip(axes, metrics):
        for data, color, lbl, ls in ((a07, C_A07, "α = 0.7（重采样）", "-"),
                                     (a00, C_A00, "α = 0（自然先验，同预算）", "--")):
            if not data["step"]:
                continue
            ax.plot(data["step"], data[key], color=color, lw=1.8, ls=ls,
                    marker="o", ms=3.2, label=lbl)
        ax.set_xlabel("step")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylim(*ylim)
        ax.set_xlim(left=0)
        ax.legend(frameon=False, fontsize=8.6, loc="best")
    axes[0].set_ylabel("指标")
    fig.suptitle("等预算消融：同数据池、同 12 000 步调度，只改省份重采样强度",
                 fontsize=12, y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"[fig] {out.relative_to(PROJECT_ROOT)}")


def table_at(a07: dict, a00: dict, step: int) -> None:
    def nearest(d):
        if not d["step"]:
            return None
        i = int(np.argmin([abs(s - step) for s in d["step"]]))
        return {k: d[k][i] for k in ("step", "acc", "macro", "wan_fp")}

    r7, r0 = nearest(a07), nearest(a00)
    print(f"\n--- 等预算对比 @ step≈{step} ---")
    print("| 模型 | step | 整牌 acc | 非皖头部宏平均 | 皖误报率 |")
    print("|---|---:|---:|---:|---:|")
    for tag, r in (("α=0.7", r7), ("α=0", r0)):
        if r:
            print(f"| {tag} | {r['step']} | {r['acc']:.1%} | {r['macro']:.1%} "
                  f"| {r['wan_fp']:.1%} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a07", default="runs/plate_bal/train_log.csv")
    ap.add_argument("--a00", default="runs/ctrl_a0/train_log.csv")
    ap.add_argument("--out", default="runs/report_figs/fig9_ablation.png")
    ap.add_argument("--step", type=int, default=12000)
    args = ap.parse_args()

    a07 = read_val(PROJECT_ROOT / args.a07)
    a00 = read_val(PROJECT_ROOT / args.a00)
    if not a00["step"]:
        print(f"[skip] no validation rows in {args.a00}")
        return 1
    fig_ablation(a07, a00, PROJECT_ROOT / args.out)
    table_at(a07, a00, args.step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
