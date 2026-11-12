"""Diagnose the detection head on a few validation images."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpr.ccpd import build_loader  # noqa: E402
from lpr.geometry import quad_to_bbox_t  # noqa: E402
from lpr.model import PlateNet  # noqa: E402
from lpr.loss import PlateLoss  # noqa: E402
from lpr.evaluate import bbox_iou  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "runs/smoke_fix/checkpoints/best.pth")
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg = ckpt["config"]["model"]
    model = PlateNet((cfg["input_h"], cfg["input_w"]), cfg["stride"],
                     (cfg["crop_h"], cfg["crop_w"])).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = ROOT / "data/CCPD2019"
    loader = build_loader(root, root / "splits/val.txt", batch_size=8, train=False,
                          input_hw=(model.input_h, model.input_w), workers=0, limit=8)
    batch = next(iter(loader))
    img = batch["image"].cuda()
    gt = batch["corners"].cuda()

    with torch.no_grad():
        out = model(img)
    obj = out["obj"].reshape(img.shape[0], -1)
    pos = PlateLoss.assign(gt, model)
    logits = obj.log()
    print("obj logit stats: min %.2f mean %.2f max %.2f" % (obj.min(), obj.mean(), obj.max()))
    print("positives per image:", pos.sum(1).tolist())
    print("obj logit on positives: mean %.2f  max %.2f" % (
        (obj * pos).sum() / pos.sum().clamp(min=1), (obj.masked_fill(pos == 0, -1e9)).max()))
    print("obj logit on negatives: mean %.2f  max %.2f" % (
        (obj * (1 - pos)).sum() / (1 - pos).sum(), (obj.masked_fill(pos == 1, -1e9)).max()))

    pred = out["corners"]
    gt_px = gt * torch.tensor([model.input_w, model.input_h], device="cuda")
    pred_px = pred * torch.tensor([model.input_w, model.input_h], device="cuda")
    print("iou:", bbox_iou(quad_to_bbox_t(pred_px), quad_to_bbox_t(gt_px)).tolist())
    print("argmax cell (row,col) per image:")
    idx = obj.argmax(1)
    gh, gw = model.grid_hw
    print([(int(i // gw), int(i % gw)) for i in idx])
    print("gt centre cell:", [((float(c[:, 1].mean()) * model.input_h) / 8, (float(c[:, 0].mean()) * model.input_w) / 8)
                              for c in gt])
    print("pred corners[0]:", pred[0].tolist())
    print("gt   corners[0]:", gt[0].tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
