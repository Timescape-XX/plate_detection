"""Evaluation suite: one checkpoint, several splits, per-province breakdown.

``train.py`` only reports aggregate accuracy on a val split -- and on a 92 %-皖
split that number is blind to the exact failure this project is fighting.  This
script measures what actually matters:

* ``test_prov``  province-uniform (31 provinces x quota) -> per-province accuracy
                 and a gt-province x pred-province confusion matrix
* ``val_base``   ccpd_base, i.e. the old training distribution -> did the 皖
                 accuracy survive the re-balancing?
* ``test_mix``   natural-prior holdout of blur/challenge/db/fn/rotate/tilt/weather
                 -> robustness on the hard subsets

It also implements the **post-hoc prior correction** (logit adjustment) that is
the inference-time dual of the re-sampling:

    z'_p = z_p - τ · log π_p ,    p ∈ province classes

with ``π`` the training-province frequency. ``τ = 1`` removes the prior exactly;
the ``--tau-sweep`` traces the whole 皖-accuracy vs non-皖-accuracy trade-off on
a single checkpoint, without retraining.

    python tools/eval_suite.py --weights runs/plate_bal/checkpoints/best.pth --name new
    python tools/eval_suite.py --weights runs/smoke_fix/checkpoints/best.pth --name old
    python tools/eval_suite.py --weights ... --splits test_prov --tau-sweep 0,0.5,1,1.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lpr.ccpd import build_loader  # noqa: E402
from lpr.charset import NUM_CLASSES, PROVINCES, ctc_greedy_decode  # noqa: E402
from lpr.evaluate import bbox_iou, levenshtein  # noqa: E402
from lpr.geometry import quad_to_bbox_t  # noqa: E402
from predict import load_model  # noqa: E402

HEAD_MIN = 50            # a province needs this many test images to enter the macro average


def province_prior(split_file: Path) -> torch.Tensor:
    """Training province frequency as a ``(NUM_CLASSES,)`` log-prior vector."""
    counts = {p: 0 for p in PROVINCES}
    lines = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for ln in lines:
        parts = Path(ln).stem.split("-")
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[4].split("_")[0])
        except (ValueError, IndexError):
            continue
        if 0 <= idx < len(PROVINCES):
            counts[PROVINCES[idx]] += 1
    total = max(sum(counts.values()), 1)
    log_pi = torch.full((NUM_CLASSES,), float("nan"))
    for i, p in enumerate(PROVINCES):
        log_pi[i] = torch.log(torch.tensor(max(counts[p] / total, 1e-6))).item()
    return log_pi


@torch.no_grad()
def eval_split(model, loader, device, tau: float = 0.0, log_pi: torch.Tensor | None = None,
               max_batches: int | None = None, iou_thr: float = 0.7) -> dict:
    model.eval()
    n = det = exact = 0
    iou_sum = char_err = 0.0
    char_cnt = 0
    groups = {}
    pair = {}          # (gt_prov, pred_prov) -> count
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        corners_gt = batch["corners"].to(device, non_blocking=True)
        out = model(image)
        logits = out["logits"]
        if tau and log_pi is not None:
            adj = torch.zeros_like(logits[0, 0])
            adj[:len(PROVINCES)] = tau * log_pi[:len(PROVINCES)].to(logits.device)
            logits = logits - adj.view(1, 1, -1)
        preds = ctc_greedy_decode(logits)

        pred_px = out["corners"] * torch.tensor([model.input_w, model.input_h],
                                                dtype=out["corners"].dtype, device=device)
        gt_px = corners_gt * torch.tensor([model.input_w, model.input_h],
                                          dtype=corners_gt.dtype, device=device)
        iou = bbox_iou(quad_to_bbox_t(pred_px), quad_to_bbox_t(gt_px))

        for k, (pred, gt) in enumerate(zip(preds, batch["texts"])):
            n += 1
            iou_sum += float(iou[k])
            det += int(float(iou[k]) >= iou_thr)
            ok = pred == gt
            exact += int(ok)
            d = levenshtein(pred, gt)
            char_err += d
            char_cnt += len(gt)
            gp = batch["provinces"][k]
            g = groups.setdefault(gp, {"n": 0, "ok": 0, "wan_fp": 0})
            g["n"] += 1
            g["ok"] += int(ok)
            pp = pred[:1] if pred else "∅"
            g["wan_fp"] += int(gp != "皖" and pp == "皖")
            pair[f"{gp}>{pp}"] = pair.get(f"{gp}>{pp}", 0) + 1

    n = max(n, 1)
    nonwan = [g for p, g in groups.items() if p != "皖"]
    head = [g["ok"] / g["n"] for g in nonwan if g["n"] >= HEAD_MIN]
    return {
        "n": n,
        "iou": iou_sum / n,
        "det_rate": det / n,
        "plate_acc": exact / n,
        "cer": char_err / max(char_cnt, 1),
        "wan_n": groups.get("皖", {}).get("n", 0),
        "wan_acc": (groups.get("皖", {}).get("ok", 0) / groups["皖"]["n"]) if groups.get("皖") else 0.0,
        "nonwan_n": sum(g["n"] for g in nonwan),
        "nonwan_acc": sum(g["ok"] for g in nonwan) / max(sum(g["n"] for g in nonwan), 1),
        "wan_fp_rate": sum(g["wan_fp"] for g in nonwan) / max(sum(g["n"] for g in nonwan), 1),
        "macro_acc_head": sum(head) / len(head) if head else 0.0,
        "n_head_provinces": len(head),
        "by_province": {p: {"n": g["n"], "acc": g["ok"] / g["n"],
                            "wan_fp": g["wan_fp"] / g["n"] if p != "皖" else 0.0}
                        for p, g in sorted(groups.items())},
        "province_pairs": dict(sorted(pair.items(), key=lambda kv: -kv[1])),
    }


def fmt(name: str, r: dict) -> str:
    return (f"{name:12s} n={r['n']:6d} acc={r['plate_acc']:.4f} iou={r['iou']:.4f} "
            f"det={r['det_rate']:.4f} cer={r['cer']:.4f} | "
            f"皖 {r['wan_acc']:.3f} | non-皖 {r['nonwan_acc']:.3f} | "
            f"macro_head {r['macro_acc_head']:.3f} | 皖FP {r['wan_fp_rate']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default=None, help="tag for the output json (default: checkpoint stem)")
    ap.add_argument("--splits", default="test_prov,val_base,test_mix",
                    help="comma separated split file stems under data/CCPD2019/splits")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tau-sweep", default=None,
                    help="e.g. 0,0.5,1,1.5 : post-hoc province prior correction on the first split")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    weights = Path(args.weights) if Path(args.weights).is_absolute() else PROJECT_ROOT / args.weights
    if not weights.is_file():
        print(f"[FATAL] checkpoint not found: {weights}")
        return 2
    model = load_model(weights, device)
    data_root = PROJECT_ROOT / "data" / "CCPD2019"
    name = args.name or weights.parent.parent.name
    results: dict[str, dict] = {}

    priors = None
    if args.tau_sweep:
        priors = province_prior(data_root / "splits" / "train_mix.txt")

    for stem in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split_file = data_root / "splits" / f"{stem}.txt"
        if not split_file.is_file():
            print(f"[skip] missing split {split_file}")
            continue
        loader = build_loader(data_root, split_file, batch_size=args.batch_size, train=False,
                             input_hw=(model.input_h, model.input_w), workers=args.workers,
                             limit=args.limit)
        r = eval_split(model, loader, device)
        results[stem] = r
        print(fmt(stem, r), flush=True)

        if args.tau_sweep:
            sweep = []
            for tau in [float(t) for t in args.tau_sweep.split(",")]:
                if tau == 0.0:
                    sweep.append({"tau": 0.0, **{k: r[k] for k in
                                                 ("plate_acc", "wan_acc", "nonwan_acc",
                                                  "wan_fp_rate", "macro_acc_head", "cer")}})
                    continue
                rt = eval_split(model, loader, device, tau=tau, log_pi=priors)
                sweep.append({"tau": tau, **{k: rt[k] for k in
                                             ("plate_acc", "wan_acc", "nonwan_acc",
                                              "wan_fp_rate", "macro_acc_head", "cer")}})
                print(f"  tau={tau:<4} acc={rt['plate_acc']:.4f} 皖={rt['wan_acc']:.3f} "
                      f"non-皖={rt['nonwan_acc']:.3f} macro={rt['macro_acc_head']:.3f} "
                      f"皖FP={rt['wan_fp_rate']:.3f}", flush=True)
            results[f"{stem}__tau_sweep"] = sweep

    out = Path(args.out) if args.out else PROJECT_ROOT / "runs" / "eval" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"weights": str(weights), "name": name, "results": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
