"""Evaluate a checkpoint on a CCPD split.

    python plate_detection/evaluate.py --weights runs/plate_bal/checkpoints/best.pth \
                                       --split test_prov --limit 5000

``--split`` is any stem under ``data/CCPD2019/splits/`` (val / test / val_base /
test_prov / test_mix / ...).  The output includes a per-province breakdown: on a
92 %-皖 split the aggregate accuracy hides exactly the failure mode this project
is about.  For a full before/after comparison use ``tools/eval_suite.py``.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from lpr.ccpd import build_loader
from lpr.evaluate import run_eval
from lpr.paths import PROJECT_ROOT, resolve_path

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
from predict import load_model  # noqa: E402  (shares the checkpoint-loading logic)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val",
                    help="split stem under data/CCPD2019/splits (val, test, val_base, test_prov, ...)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    weights = resolve_path(args.weights)
    if not weights.is_file():
        print(f"[FATAL] checkpoint not found: {args.weights} (tried {weights})")
        return 2
    model = load_model(weights, device)
    data_root = ROOT / "data" / "CCPD2019"
    split_file = data_root / "splits" / f"{args.split}.txt"
    if not split_file.is_file():
        print(f"[FATAL] split file not found: {split_file}\n"
              f"        expected CCPD2019 under {data_root} (splits/{{train,val,test}}.txt "
              f"+ image subdirectories)")
        return 2
    loader = build_loader(data_root, split_file,
                          batch_size=args.batch_size, train=False,
                          input_hw=(model.input_h, model.input_w), workers=args.workers,
                          limit=args.limit)
    metrics = run_eval(model, loader, device, max_batches=args.max_batches)
    print(json.dumps({k: v for k, v in metrics.items() if k != "examples"},
                     indent=2, ensure_ascii=False))
    print("examples:", [(e["pred"], e["gt"], round(e["iou"], 3)) for e in metrics["examples"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
