"""Build every figure used by REPORT.md from the run artefacts.

    python tools/make_figures.py                       # all figures
    python tools/make_figures.py --run runs/plate_bal  # explicit run directory

Inputs (produced by train.py / eval_suite.py / audit_dataset.py)
----------------------------------------------------------------
``<run>/train_log.csv``         step-level loss + val metrics
``<run>/sampling_plan.json``    natural vs effective province distribution
``<run>/val_history.json``      per-validation province breakdown
``runs/eval/<name>.json``       eval_suite output for any checkpoint
``runs/audit/dataset_audit.json``, ``runs/audit/split_stats.json``

Outputs -> ``runs/report_figs/*.png`` (150 dpi, white background, CJK font)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "runs" / "report_figs"

C_INK = "#1b1f24"
C_MUTED = "#6b7280"
C_GRID = "#e5e7eb"
C_WAN = "#c0392b"        # 皖 -- the over-predicted class
C_OTHER = "#2f6fd0"
C_ACC = "#0f8a5f"
C_OLD = "#9aa4af"
C_NEW = "#2f6fd0"

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


# ------------------------------------------------------------------ data loading
def read_log(path: Path) -> dict:
    steps, loss = [], []
    comp = {k: [] for k in ("cls", "giou", "corner", "ctc", "ctc_gt")}
    comp_steps = {k: [] for k in comp}
    lr_s, lr_v = [], []
    val = {"step": [], "acc": [], "iou": [], "det": [], "cer": [],
           "wan_fp": [], "macro": []}
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("val_iou"):
                val["step"].append(int(row["step"]))
                val["acc"].append(float(row["val_plate_acc"]))
                val["iou"].append(float(row["val_iou"]))
                val["det"].append(float(row["val_det_rate"]))
                val["cer"].append(float(row["val_cer"]))
                val["wan_fp"].append(float(row.get("val_wan_fp") or 0))
                val["macro"].append(float(row.get("val_macro_head") or 0))
            elif row.get("loss"):
                steps.append(int(row["step"]))
                loss.append(float(row["loss"]))
                for k in comp:
                    v = row.get(k)
                    if v not in (None, ""):
                        comp[k].append(float(v))
                        comp_steps[k].append(int(row["step"]))
                lr_s.append(int(row["step"]))
                lr_v.append(float(row["lr"]))
    return {"step": steps, "loss": loss, "comp": comp, "comp_step": comp_steps,
            "lr_step": lr_s, "lr": lr_v, "val": val}


def smooth(y, k: int = 5):
    if len(y) < k or k < 2:
        return np.asarray(y, dtype=float)
    ker = np.ones(k) / k
    pad = k // 2
    yp = np.pad(np.asarray(y, dtype=float), pad, mode="edge")
    return np.convolve(yp, ker, mode="valid")[:len(y)]


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] {out.relative_to(PROJECT_ROOT)}")
    return out


# -------------------------------------------------------------------- fig: data
def fig_dataset(audit: dict, split_stats: dict) -> None:
    prov = audit["global_provinces"]
    order = sorted(prov.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in order]
    vals = [v for _, v in order]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    colors = [C_WAN if n == "皖" else C_OTHER for n in names]
    bars = ax.bar(range(len(names)), vals, color=colors, width=0.75, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("图片数 (log)")
    ax.set_title("CCPD2019 各省份样本量：92.8% 是「皖」", fontsize=11, pad=10)
    for b, v, nm in zip(bars, vals, names):
        if nm in ("皖", "苏", "浙", "藏", "宁", "琼", "青"):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.35, f"{v:,}", ha="center",
                    fontsize=8, color=C_WAN if nm == "皖" else C_MUTED, zorder=5)
    ax.set_ylim(1, max(vals) * 8)
    ax.grid(axis="x", visible=False)

    # split composition
    ax2 = axes[1]
    labels = ["train_mix\n(训练)", "val_base\n(皖主导验证)", "val_mix\n(混合验证)",
              "test_prov\n(省份均衡)", "test_mix\n(自然先验测试)"]
    keys = ["train_mix.txt", "val_base.txt", "val_mix.txt", "test_prov.txt", "test_mix.txt"]
    wan = [split_stats.get(k, {}).get("wan_ratio", 0) * 100 for k in keys]
    non = [100 - w for w in wan]
    y = np.arange(len(keys))
    ax2.barh(y, wan, color=C_WAN, label="皖", height=0.62, zorder=3)
    ax2.barh(y, non, left=wan, color=C_OTHER, label="非皖（30 省）", height=0.62, zorder=3)
    for i, (w, k) in enumerate(zip(wan, keys)):
        n = split_stats.get(k, {}).get("n", 0)
        ax2.text(1.5, i, f"{w:.1f}%", va="center", fontsize=9, color="white",
                 fontweight="bold", zorder=5)
        ax2.text(101.5, i, f"n={n:,}", va="center", fontsize=8.5, color=C_MUTED)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlim(0, 126)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.invert_yaxis()
    ax2.set_xlabel("占比 (%)")
    ax2.set_title("五个 split 的省份构成", fontsize=11, pad=10)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=2,
               fontsize=9, frameon=False)
    ax2.grid(axis="y", visible=False)
    save(fig, "fig1_dataset_province_skew.png")


# ---------------------------------------------------------------- fig: sampling
def fig_sampling(plan: dict, top: int = 16) -> None:
    nat = plan["province_natural"]
    eff = plan["province_effective"]
    names = sorted(nat, key=lambda k: -nat[k])[:top]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0), sharey=False)
    for ax, data, title, cmap, logx in (
        (axes[0], nat, "训练池的真实分布（对数轴）", C_MUTED, True),
        (axes[1], eff, f"重采样后的有效分布 (α={plan['alpha']})", C_NEW, False),
    ):
        vals = [max(data.get(n, 0) * 100, 1e-3) for n in names]
        colors = [C_WAN if n == "皖" else cmap for n in names]
        if logx:
            # bars on a log axis misrepresent area -> use lollipops instead
            ax.hlines(np.arange(len(names)), 3e-3, vals, color=colors, lw=1.6, alpha=0.6, zorder=3)
            ax.scatter(vals, np.arange(len(names)), s=42, color=colors, zorder=4)
        else:
            ax.barh(np.arange(len(names)), vals, color=colors, height=0.68, zorder=3)
        for i, v in enumerate(vals):
            ax.text(v * 1.18 if logx else v + max(vals) * 0.015, i,
                    f"{data.get(names[i], 0) * 100:.2f}%", va="center", fontsize=8, color=C_MUTED)
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(3e-3, 400)
        else:
            ax.set_xlim(0, max(vals) * 1.18)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("采样概率 (%)")
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("省份均衡重采样：皖 92.7% → 19.8%，30 个非皖省份合计 80.2%",
                 fontsize=12, y=1.02)
    save(fig, "fig2_sampling_effect.png")


# -------------------------------------------------------------------- fig: loss
def fig_loss(log: dict, run_label: str = "") -> None:
    """Three panels: weighted decomposition, per-component (log y), LR schedule."""
    # weights from the config -> the stack sums exactly to the logged total
    W = {"cls": 1.0, "giou": 1.0, "corner": 0.5, "ctc": 1.0, "ctc_gt": 0.5}
    NAME = {"cls": "objectness (focal)", "giou": "GIoU", "corner": "corner L1",
            "ctc": "CTC (pred crop)", "ctc_gt": "CTC (GT crop)"}
    COL = {"cls": C_OTHER, "giou": "#8e44ad", "corner": "#e67e22",
           "ctc": C_ACC, "ctc_gt": "#0b5d3f"}

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.1),
                             gridspec_kw={"width_ratios": [1.15, 1.05, 0.8]})
    if not log:
        return

    # ---- (a) stacked weighted contributions -------------------------------
    ax = axes[0]
    st = log["comp_step"]["corner"]
    stack = [smooth(log["comp"][k], 9) * W[k] for k in ("ctc_gt", "corner", "giou", "cls", "ctc")]
    order = ["ctc_gt", "corner", "giou", "cls", "ctc"]
    ax.stackplot(st, *stack, labels=[NAME[k] for k in order],
                 colors=[COL[k] for k in order], alpha=0.85, linewidth=0)
    ax.plot(log["step"], smooth(log["loss"], 9), color=C_INK, lw=1.8,
            label="total (实测)", zorder=5)
    ax.set_xlabel("step")
    ax.set_ylabel("加权损失贡献")
    ax.set_title("(a) 损失分解：堆叠面积 = total", fontsize=11, pad=8)
    ax.set_xlim(0, st[-1])
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8.2, loc="upper right", ncol=1,
              handlelength=1.4, labelspacing=0.32)

    # ---- (b) per-component decay on log scale ----------------------------
    ax = axes[1]
    for k in ("cls", "giou", "corner", "ctc", "ctc_gt"):
        y = np.maximum(smooth(log["comp"][k], 9), 1e-3)
        ax.plot(log["comp_step"][k], y, color=COL[k], lw=1.3, label=NAME[k])
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss（log，未加权）")
    ax.set_title("(b) 各分量收敛速度", fontsize=11, pad=8)
    ax.set_xlim(0, st[-1])
    ax.legend(frameon=False, fontsize=8.2, loc="upper right", handlelength=1.4,
              labelspacing=0.32)

    # ---- (c) LR schedule -------------------------------------------------
    ax = axes[2]
    ax.plot(log["lr_step"], log["lr"], color="#2c7fb8", lw=1.8)
    ax.fill_between(log["lr_step"], 0, log["lr"], color="#2c7fb8", alpha=0.12)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("(c) warmup + cosine", fontsize=11, pad=8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, max(log["lr"]) * 1.15)
    wu = int(np.argmax(log["lr"]))
    ax.axvline(log["lr_step"][wu], color=C_MUTED, ls=":", lw=1.0)
    ax.annotate(f"峰值 {max(log['lr']):.4f}\n@ step {log['lr_step'][wu]}",
                xy=(log["lr_step"][wu], max(log["lr"])),
                xytext=(log["lr_step"][wu] + st[-1] * 0.10, max(log["lr"]) * 0.62),
                fontsize=8.5, color=C_MUTED,
                arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.0))
    fig.suptitle(f"训练损失曲线 {run_label}".strip(), fontsize=12, y=1.01)
    save(fig, "fig3_loss_curves.png")


# --------------------------------------------------------------- fig: val curve
def fig_val(log: dict) -> None:
    v = log.get("val") or {}
    if not v or not v["step"]:
        print("[skip] no validation rows yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0))
    ax = axes[0]
    ax.plot(v["step"], v["acc"], color=C_NEW, lw=1.8, marker="o", ms=3, label="端到端准确率")
    ax.plot(v["step"], v["iou"], color="#e67e22", lw=1.4, marker=".", ms=3, label="检测 IoU")
    ax.plot(v["step"], v["det"], color="#8e44ad", lw=1.2, ls="--", label="检出率 (IoU≥0.7)")
    ax.set_xlabel("step")
    ax.set_ylabel("指标")
    ax.set_ylim(0, 1.02)
    ax.set_title("验证集（val_mix）", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_xlim(left=0)

    ax = axes[1]
    ax.plot(v["step"], v["macro"], color=C_ACC, lw=1.8, marker="o", ms=3,
            label="非皖头部省份宏平均准确率")
    ax.plot(v["step"], v["wan_fp"], color=C_WAN, lw=1.6, marker="s", ms=3,
            label="非皖被判成皖的比例")
    ax.plot(v["step"], v["cer"], color=C_MUTED, lw=1.2, ls=":", label="CER")
    ax.set_xlabel("step")
    ax.set_ylabel("比例")
    ax.set_ylim(0, 1.02)
    ax.set_title("省份维度的健康度", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(left=0)
    save(fig, "fig4_val_curves.png")


# ------------------------------------------------------ fig: per-province acc
def fig_province_acc(old: dict, new: dict, split: str = "test_prov", min_n: int = 20) -> None:
    def get(pkg):
        r = pkg["results"].get(split)
        return r["by_province"] if r else {}

    o, n = get(old), get(new)
    names = [p for p in n if n[p]["n"] >= min_n]
    if not names:
        print("[skip] province accuracy: not enough samples")
        return
    names.sort(key=lambda p: -n[p]["n"])
    x = np.arange(len(names))
    wo = [o.get(p, {}).get("acc", 0) for p in names]
    wn = [n.get(p, {}).get("acc", 0) for p in names]

    fig, ax = plt.subplots(figsize=(12.4, 4.4))
    ax.bar(x - 0.2, wo, 0.4, color=C_OLD, label=old["name"] + "（重训前）")
    ax.bar(x + 0.2, wn, 0.4, color=C_NEW, label=new["name"] + "（重训后）")
    for i, (a, b) in enumerate(zip(wo, wn)):
        # the old model sits at ~0 for almost every province: only annotate
        # the non-trivial ones, otherwise 28 stacked "0%" labels swamp the plot
        if a >= 0.03:
            ax.text(i - 0.2, a + 0.012, f"{a:.0%}", ha="center", fontsize=7.4,
                    color=C_MUTED, rotation=90, va="bottom")
        ax.text(i + 0.2, b + 0.012, f"{b:.0%}", ha="center", fontsize=7.4,
                color=C_INK, rotation=90, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n{n[p]['n']}" for p in names], fontsize=10)
    ax.set_xlabel(f"省份 / 该省测试样本数　（灰色柱未标数字者均 <3%，"
                  f"共 {sum(1 for a in wo if a < 0.03)} 个省）",
                  fontsize=8.6, color=C_MUTED, labelpad=6)
    ax.set_ylabel("整牌完全正确率")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"逐省准确率对比（{split}，每省 n≥{min_n}）", fontsize=11, pad=26)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    save(fig, "fig5_per_province_acc.png")


# ------------------------------------------------------------- fig: 皖 FP rate
def fig_wan_fp(old: dict, new: dict) -> None:
    splits = [("val_base", "val_base\n皖主导"), ("test_mix", "test_mix\n自然先验"),
              ("test_prov", "test_prov\n省份均衡")]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    x = np.arange(len(splits))
    wo = [old["results"].get(s, {}).get("wan_fp_rate", 0) for s, _ in splits]
    wn = [new["results"].get(s, {}).get("wan_fp_rate", 0) for s, _ in splits]
    ax.bar(x - 0.2, wo, 0.4, color=C_OLD, label="重训前")
    ax.bar(x + 0.2, wn, 0.4, color=C_NEW, label="重训后")
    for i, (a, b) in enumerate(zip(wo, wn)):
        ax.text(i - 0.2, a + 0.02, f"{a:.1%}", ha="center", fontsize=8.5, color=C_MUTED)
        ax.text(i + 0.2, b + 0.02, f"{b:.1%}", ha="center", fontsize=8.5, color=C_INK)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in splits], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("非皖车牌被判成「皖」的比例")
    ax.set_title("皖 误报率", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", visible=False)

    # non-皖 vs 皖 accuracy, both models
    ax = axes[1]
    r_o, r_n = old["results"].get("test_prov", {}), new["results"].get("test_prov", {})
    cats = ["皖\n准确率", "非皖\n准确率", "非皖头部\n宏平均", "整牌\n准确率"]
    keys = ["wan_acc", "nonwan_acc", "macro_acc_head", "plate_acc"]
    vo = [r_o.get(k, 0) for k in keys]
    vn = [r_n.get(k, 0) for k in keys]
    x = np.arange(len(cats))
    ax.bar(x - 0.2, vo, 0.4, color=C_OLD, label="重训前")
    ax.bar(x + 0.2, vn, 0.4, color=C_NEW, label="重训后")
    for i, (a, b) in enumerate(zip(vo, vn)):
        ax.text(i - 0.2, a + 0.02, f"{a:.1%}", ha="center", fontsize=8.5, color=C_MUTED)
        ax.text(i + 0.2, b + 0.02, f"{b:.1%}", ha="center", fontsize=8.5, color=C_INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("准确率")
    ax.set_title("test_prov 上四个视角", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "fig6_wan_confusion.png")


# ------------------------------------------------------- fig: per-slot analysis
def fig_slots(old: dict, new: dict) -> None:
    """Which character slot is still the bottleneck, and what confuses slot 1?"""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})

    # (a) per-slot accuracy
    ax = axes[0]
    slots = sorted(old["slot_acc"], key=int)
    x = np.arange(len(slots))
    so = [old["slot_acc"][s] for s in slots]
    sn = [new["slot_acc"][s] for s in slots]
    ax.bar(x - 0.2, so, 0.4, color=C_OLD, label=f"{old['name']}（重训前）")
    ax.bar(x + 0.2, sn, 0.4, color=C_NEW, label=f"{new['name']}（重训后）")
    for i, b in enumerate(sn):
        ax.text(i + 0.2, b + 0.015, f"{b:.0%}", ha="center", fontsize=7.8, color=C_INK)
    for i, a in enumerate(so):
        ax.text(i - 0.2, a + 0.015, f"{a:.0%}", ha="center", fontsize=7.2, color=C_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"第{s}位" + ("\n省份" if i == 0 else "") for i, s in enumerate(slots)],
                       fontsize=9.5)
    ax.set_ylabel("该位字符准确率")
    ax.set_ylim(0, 1.14)
    ax.set_title("(a) 逐位字符准确率（test_prov）：省份位 7% → 88%", fontsize=11, pad=26)
    ax.legend(frameon=False, fontsize=8.6, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)

    # (b) slot-1 confusions: old collapses, new is diffuse
    ax = axes[1]
    rows = [(old, C_OLD, "重训前"), (new, C_NEW, "重训后")]
    labels, vals, colors = [], [], []
    k = 6
    for pkg, color, tag in rows:
        for c in pkg["top_confusions"][:k]:
            labels.append(f"{c['gt']}→{c['pred']}")
            vals.append(c["n"])
            colors.append(color)
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.72)
    for yy, v in zip(y, vals):
        ax.text(v + max(vals) * 0.015, yy, f"{v}", va="center", fontsize=8.4, color=C_INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlabel("该错误发生的次数　（上 6 行 = 重训前，全部指向「皖」；下 6 行 = 重训后，分散）",
                  fontsize=8.4, color=C_MUTED)
    ax.set_title("(b) 省份位的高频混淆对", fontsize=11, pad=26)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.14)
    ax.axhline(k - 0.5, color=C_MUTED, lw=1.1, ls="--")
    save(fig, "fig7_slot_analysis.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/plate_bal")
    ap.add_argument("--old-eval", default="runs/eval/old_smoke.json")
    ap.add_argument("--new-eval", default="runs/eval/new_bal.json")
    args = ap.parse_args()

    run = PROJECT_ROOT / args.run
    audit = json.loads((PROJECT_ROOT / "runs/audit/dataset_audit.json").read_text(encoding="utf-8"))
    splits = json.loads((PROJECT_ROOT / "runs/audit/split_stats.json").read_text(encoding="utf-8"))
    fig_dataset(audit, splits)

    plan_path = run / "sampling_plan.json"
    if plan_path.is_file():
        fig_sampling(json.loads(plan_path.read_text(encoding="utf-8")))

    log = read_log(run / "train_log.csv")
    if log:
        fig_loss(log)
        fig_val(log)
    else:
        print("[skip] no train_log.csv yet")

    old_p, new_p = PROJECT_ROOT / args.old_eval, PROJECT_ROOT / args.new_eval
    if old_p.is_file() and new_p.is_file():
        old = json.loads(old_p.read_text(encoding="utf-8"))
        new = json.loads(new_p.read_text(encoding="utf-8"))
        fig_province_acc(old, new)
        fig_wan_fp(old, new)
    else:
        print(f"[skip] comparison figures need both {old_p.name} and {new_p.name}")

    pos_old = PROJECT_ROOT / "runs/audit/pos_old_smoke.json"
    pos_new = PROJECT_ROOT / "runs/audit/pos_new_bal.json"
    if pos_old.is_file() and pos_new.is_file():
        fig_slots(json.loads(pos_old.read_text(encoding="utf-8")),
                  json.loads(pos_new.read_text(encoding="utf-8")))
    else:
        print("[skip] slot figure needs runs/audit/pos_*.json (tools/pos_analysis.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
