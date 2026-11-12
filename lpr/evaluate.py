"""Evaluation helpers shared by ``train.py`` and ``evaluate.py``."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .charset import ctc_greedy_decode
from .geometry import quad_to_bbox_t, warp_quads


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def bbox_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)
    iw = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(min=0)
    ih = (torch.min(ay2, by2) - torch.max(ay1, by1)).clamp(min=0)
    inter = iw * ih
    ua = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    ub = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    return inter / (ua + ub - inter + eps)


@torch.no_grad()
def run_eval(model, loader, device, max_batches: int | None = None, iou_thr: float = 0.7,
             by_province: bool = True) -> Dict[str, float]:
    """Returns detection IoU / recall, end-to-end plate accuracy and CER.

    With ``by_province`` the result also carries a per-province breakdown, which
    is what the 皖-collapse analysis needs: an aggregate accuracy over a 92 %-皖
    set says nothing about whether the recogniser has actually learned the other
    30 province glyphs.
    """
    model.eval()
    n = 0
    iou_sum = 0.0
    det_hits = 0
    exact = 0
    char_err = 0.0
    char_cnt = 0
    text_ok_when_detected = 0
    examples: List[dict] = []
    groups: Dict[str, Dict[str, int]] = {}
    wan_fp = 0            # gt province != 皖 but pred province == 皖
    nonwan_n = 0

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        corners_gt = batch["corners"].to(device, non_blocking=True)
        out = model(image)

        pred_px = out["corners"] * torch.tensor(
            [model.input_w, model.input_h], dtype=out["corners"].dtype, device=device
        )
        gt_px = corners_gt * torch.tensor(
            [model.input_w, model.input_h], dtype=corners_gt.dtype, device=device
        )
        iou = bbox_iou(quad_to_bbox_t(pred_px), quad_to_bbox_t(gt_px))

        preds = ctc_greedy_decode(out["logits"])
        b = image.shape[0]
        n += b
        iou_sum += float(iou.sum())
        det_hits += int((iou >= iou_thr).sum())
        provs = batch.get("provinces") or [None] * b
        for k, (pred, gt) in enumerate(zip(preds, batch["texts"])):
            ok = pred == gt
            exact += int(ok)
            d = levenshtein(pred, gt)
            char_err += d
            char_cnt += len(gt)
            if ok and float(iou[k]) >= iou_thr:
                text_ok_when_detected += 1
            if by_province:
                g = groups.setdefault(provs[k] or "?", {"n": 0, "ok": 0, "wan_fp": 0})
                g["n"] += 1
                g["ok"] += int(ok)
                gt_prov = gt[:1] if gt else "?"
                pred_prov = pred[:1] if pred else "∅"
                if gt_prov != "皖":
                    nonwan_n += 1
                    is_fp = pred_prov == "皖"
                    g["wan_fp"] += int(is_fp)
                    wan_fp += int(is_fp)
            if len(examples) < 8:
                examples.append({"pred": pred, "gt": gt, "iou": float(iou[k])})

    n = max(n, 1)
    model.train()
    result = {
        "n": n,
        "iou": iou_sum / n,
        "det_rate": det_hits / n,
        f"det@{iou_thr:.2f}": det_hits / n,
        "plate_acc": exact / n,
        "plate_acc_when_detected": text_ok_when_detected / max(det_hits, 1),
        "cer": char_err / max(char_cnt, 1),
        "examples": examples,
    }
    if by_province:
        result["by_province"] = {
            p: {"n": g["n"], "acc": g["ok"] / g["n"]} for p, g in sorted(groups.items())
        }
        result["province_err"] = {
            p: g["wan_fp"] / g["n"] for p, g in sorted(groups.items()) if p != "皖"
        }
        result["wan_fp_rate"] = wan_fp / max(nonwan_n, 1)
        result["nonwan_n"] = nonwan_n
        head = [g["ok"] / g["n"] for p, g in groups.items() if p != "皖" and g["n"] >= 50]
        result["macro_acc_nonwan_head"] = sum(head) / len(head) if head else 0.0
    return result


@torch.no_grad()
def predict_batch(model, image: torch.Tensor) -> Dict:
    out = model(image)
    return {
        "corners": out["corners"],
        "scores": out["scores"],
        "texts": ctc_greedy_decode(out["logits"]),
        "crop": warp_quads(
            image,
            out["corners"] * torch.tensor([model.input_w, model.input_h], device=image.device),
            model.crop_hw,
        ),
    }
