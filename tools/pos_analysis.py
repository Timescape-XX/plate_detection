"""Where do the remaining errors come from? Per-character-slot analysis.

The province character is slot 1 of a CCPD plate; slots 2..7/8 are a letter plus
digits. If the model is "fixed", slot 1 should stop being the bottleneck. This
script measures, on a chosen split:

* per-slot character accuracy (aligned by index, so a length mismatch counts as
  a miss on every slot past the shorter string),
* the top confusion pairs on slot 1,
* accuracy split by plate length (7 vs 8 characters, i.e. ordinary vs new-energy).

    python tools/pos_analysis.py --weights runs/plate_bal/checkpoints/best.pth \\
                                 --split test_prov --name new_bal
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lpr.ccpd import build_loader  # noqa: E402
from lpr.charset import ctc_greedy_decode  # noqa: E402
from predict import load_model  # noqa: E402

OUT_DIR = PROJECT_ROOT / "runs" / "audit"


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test_prov")
    ap.add_argument("--name", default=None)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    name = args.name or Path(args.weights).parent.parent.name
    device = torch.device(args.device)
    model = load_model(args.weights, device)
    root = PROJECT_ROOT / "data" / "CCPD2019"
    loader = build_loader(root, root / "splits" / f"{args.split}.txt", batch_size=args.batch_size,
                          train=False, input_hw=(model.input_h, model.input_w),
                          workers=args.workers, limit=args.limit)

    slot_n = collections.Counter()      # slot -> #samples that have this slot
    slot_ok = collections.Counter()     # slot -> #correct
    prov_pair = collections.Counter()   # (gt_char, pred_char) on slot 1 when wrong
    prov_n = prov_ok = 0
    by_len = collections.defaultdict(lambda: [0, 0])   # len -> [n, exact]
    exact = 0
    n = 0
    conf_by_prov = collections.Counter()      # gt prov -> wrong count
    prov_total = collections.Counter()        # gt prov -> count

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        out = model(image)
        preds = ctc_greedy_decode(out["logits"])
        for pred, gt in zip(preds, batch["texts"]):
            n += 1
            by_len[len(gt)][0] += 1
            ok = pred == gt
            exact += int(ok)
            by_len[len(gt)][1] += int(ok)
            prov_total[gt[0]] += 1
            for i, g in enumerate(gt):
                slot_n[i] += 1
                if i < len(pred) and pred[i] == g:
                    slot_ok[i] += 1
            if pred and pred[0] == gt[0]:
                prov_ok += 1
            else:
                conf_by_prov[gt[0]] += 1
                prov_pair[(gt[0], pred[0] if pred else "∅")] += 1

    slots = sorted(slot_n)
    payload = {
        "name": name,
        "weights": str(args.weights),
        "split": args.split,
        "n": n,
        "exact_acc": exact / max(n, 1),
        "slot_acc": {str(i + 1): slot_ok[i] / max(slot_n[i], 1) for i in slots},
        "slot_n": {str(i + 1): slot_n[i] for i in slots},
        "province_acc": prov_ok / max(n, 1),
        "province_n": {p: prov_total[p] for p in prov_total},
        "province_err": {p: conf_by_prov[p] for p in conf_by_prov},
        "top_confusions": [
            {"gt": g, "pred": p, "n": c} for (g, p), c in prov_pair.most_common(12)
        ],
        "by_length": {str(k): {"n": v[0], "exact": v[1] / max(v[0], 1)}
                      for k, v in sorted(by_len.items())},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"pos_{name}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{name}] {args.split} n={n} exact={payload['exact_acc']:.4f} "
          f"slot1(省)={payload['province_acc']:.4f}")
    print("  逐位准确率: " + "  ".join(
        f"pos{i+1}={payload['slot_acc'][str(i+1)]:.3f}" for i in slots))
    print("  省份位 top 混淆: " + ", ".join(
        f"{c['gt']}→{c['pred']}({c['n']})" for c in payload["top_confusions"][:6]))
    print(f"[done] {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
