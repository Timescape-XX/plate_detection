"""Emit the report's markdown tables straight from the JSON artefacts.

Hand-copying numbers into a report is how reports end up wrong.  Everything below
is generated from ``runs/eval/*.json`` + ``runs/*/train_summary.json``.

    python tools/make_report_tables.py --new runs/eval/new_bal.json --old runs/eval/old_smoke.json
    # -> runs/report_figs/tables.md  (paste into REPORT.md)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DESC = {
    "test_prov": "省份均衡（核心）",
    "val_base": "ccpd_base 自然先验",
    "test_mix": "非 base 池自然先验",
}


def pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}%"


def table_main(old: dict, new: dict, ctrl: dict | None) -> str:
    cols = ["split", "模型", "整牌 acc", "皖 acc", "非皖 acc", "非皖头部宏平均", "皖误报率", "CER", "IoU", "检出率"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for split in ("test_prov", "val_base", "test_mix"):
        for pkg in (old, new) + ((ctrl,) if ctrl else ()):
            r = pkg["results"].get(split)
            if not r:
                continue
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                SPLIT_DESC.get(split, split), pkg["name"],
                pct(r["plate_acc"]), pct(r["wan_acc"]), pct(r["nonwan_acc"]),
                pct(r["macro_acc_head"]), pct(r["wan_fp_rate"]), pct(r["cer"], 2),
                f"{r['iou']:.3f}", pct(r["det_rate"])))
    return "\n".join(lines)


def table_province(old: dict, new: dict, ctrl: dict | None, min_n: int = 50) -> str:
    o = old["results"]["test_prov"]["by_province"]
    n = new["results"]["test_prov"]["by_province"]
    c = (ctrl or {}).get("results", {}).get("test_prov", {}).get("by_province", {})
    names = [p for p in n if n[p]["n"] >= min_n]
    names.sort(key=lambda p: -n[p]["n"])
    head = ["省份", "测试数", "重训前 acc", "重训前 →皖", "重训后 acc", "重训后 →皖"]
    if c:
        head.insert(4, "α=0 对照 acc")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for p in names:
        row = [p, str(n[p]["n"]), pct(o.get(p, {}).get("acc", 0)),
               pct(o.get(p, {}).get("wan_fp", 0)), pct(n[p]["acc"]), pct(n[p]["wan_fp"])]
        if c:
            row.insert(4, pct(c.get(p, {}).get("acc", 0)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def table_tau(pkg: dict, split: str = "test_prov") -> str:
    sweep = pkg["results"].get(f"{split}__tau_sweep")
    if not sweep:
        return ""
    head = ["τ", "整牌 acc", "皖 acc", "非皖 acc", "非皖头部宏平均", "皖误报率"]
    lines = [f"**{pkg['name']}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in sweep:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            f"{s['tau']:g}", pct(s["plate_acc"]), pct(s["wan_acc"]), pct(s["nonwan_acc"]),
            pct(s["macro_acc_head"]), pct(s["wan_fp_rate"])))
    return "\n".join(lines)


def table_train(run: Path) -> str:
    p = run / "train_summary.json"
    if not p.is_file():
        return ""
    s = json.loads(p.read_text(encoding="utf-8"))
    rows = [
        ("优化步数", f"{s['steps']:,}"),
        ("样本吞吐", f"{s['steps'] * s['batch_size']:,} 张（≈ {s['steps'] * s['batch_size'] / 215179:.1f} epoch）"),
        ("总耗时", f"{s['elapsed_sec'] / 3600:.2f} h"),
        ("单步耗时", f"{s['sec_per_step'] * 1000:.0f} ms（batch={s['batch_size']}）"),
        ("吞吐", f"{s['batch_size'] / s['sec_per_step']:.0f} img/s"),
        ("峰值显存（allocated / reserved）", f"{s['peak_allocated_mb']:.0f} / {s['peak_reserved_mb']:.0f} MB"),
    ]
    lines = ["| 项目 | 值 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="runs/eval/old_smoke.json")
    ap.add_argument("--new", default="runs/eval/new_bal.json")
    ap.add_argument("--ctrl", default="runs/eval/ctrl_a0.json")
    ap.add_argument("--run", default="runs/plate_bal")
    ap.add_argument("--out", default="runs/report_figs/tables.md")
    args = ap.parse_args()

    def load(p):
        p = PROJECT_ROOT / p
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    old, new = load(args.old), load(args.new)
    ctrl = load(args.ctrl)
    if not old or not new:
        print("[FATAL] need at least the old and new eval json")
        return 2

    blocks = [
        ("主对比（§2.5）", table_main(old, new, ctrl)),
        ("逐省（§2.6）", table_province(old, new, ctrl)),
        ("先验校正扫描（§2.8）", table_tau(old) + "\n\n" + table_tau(new)),
        ("训练开销（§2.4）", table_train(PROJECT_ROOT / args.run)),
    ]
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(f"## {h}\n\n{b}" for h, b in blocks if b) + "\n", encoding="utf-8")
    print(f"[done] {out.relative_to(PROJECT_ROOT)}")
    for h, b in blocks:
        if b:
            print(f"\n--- {h} ---\n{b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
