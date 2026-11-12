"""Joint detection + recognition loss for :class:`lpr.model.PlateNet`.

Detection: FCOS-style assignment (every cell whose centre lies inside the GT
quad is positive) with focal objectness, L1 corner regression and GIoU on the
axis-aligned box. Recognition: CTC on the rectified crop, applied both to the
*GT* crop (teacher forcing, fast convergence) and to the *predicted* crop
(end-to-end alignment of detector and recogniser).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .charset import BLANK_INDEX
from .geometry import bbox_giou, points_in_quad, quad_to_bbox_t, warp_quads


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25,
                       gamma: float = 2.0, reduction: str = "sum") -> torch.Tensor:
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class PlateLoss(nn.Module):
    def __init__(
        self,
        cls_weight: float = 1.0,
        giou_weight: float = 1.0,
        corner_weight: float = 0.5,
        ctc_weight: float = 1.0,
        ctc_gt_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.giou_weight = giou_weight
        self.corner_weight = corner_weight
        self.ctc_weight = ctc_weight
        self.ctc_gt_weight = ctc_gt_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ctc = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True, reduction="mean")

    # ------------------------------------------------------------------ targets
    @staticmethod
    def assign(corners_gt: torch.Tensor, model) -> torch.Tensor:
        """``(B, 4, 2)`` normalised GT corners -> ``(B, N)`` float positive mask."""
        b = corners_gt.shape[0]
        quad_px = corners_gt * torch.tensor(
            [model.input_w, model.input_h], dtype=corners_gt.dtype, device=corners_gt.device
        )
        centers = model.cell_centers.reshape(-1, 2)               # (N,2)
        mask = torch.stack([points_in_quad(centers, quad_px[i]) for i in range(b)], dim=0)
        return mask.float()

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict, model) -> Dict[str, torch.Tensor]:
        corners_gt = batch["corners"]                            # (B,4,2) normalised
        b = corners_gt.shape[0]
        gh, gw = model.grid_hw
        stride = model.stride
        device = corners_gt.device

        pos = self.assign(corners_gt, model).to(device)          # (B, N)
        num_pos = pos.sum().clamp(min=1.0)

        # ---------------------------------------------------------- objectness
        obj = outputs["obj"].reshape(b, -1)
        loss_cls = sigmoid_focal_loss(obj, pos, self.focal_alpha, self.focal_gamma) / num_pos

        # ------------------------------------------------------------- corners
        # same helper the decoder uses -> loss and inference cannot diverge
        corners_pred = model.dense_corners(outputs["delta"])     # (B,N,4,2) pixels

        gt_px = corners_gt * torch.tensor(
            [model.input_w, model.input_h], dtype=corners_gt.dtype, device=device
        )

        loss_corner = F.l1_loss(corners_pred / stride, gt_px.unsqueeze(1).expand_as(corners_pred) / stride,
                                reduction="none")
        loss_corner = loss_corner.sum(dim=(-1, -2)) * pos
        loss_corner = loss_corner.sum() / num_pos

        giou = bbox_giou(
            quad_to_bbox_t(corners_pred).reshape(-1, 4),
            quad_to_bbox_t(gt_px).unsqueeze(1).expand(-1, gh * gw, -1).reshape(-1, 4),
        ).reshape(b, -1)
        loss_giou = ((1 - giou) * pos).sum() / num_pos

        # -------------------------------------------------------- recognition
        logits = outputs["logits"]                               # (B,T,C)
        t = logits.shape[1]
        log_probs = logits.permute(1, 0, 2).log_softmax(dim=-1)  # (T,B,C)
        input_lengths = torch.full((b,), t, dtype=torch.long, device=device)
        loss_ctc = self.ctc(log_probs, batch["targets"], input_lengths, batch["target_lengths"])

        loss_ctc_gt = torch.zeros((), device=device)
        if self.ctc_gt_weight > 0:
            gt_crop = warp_quads(batch["image"], gt_px, model.crop_hw)
            logits_gt = model.rec_head(gt_crop)
            log_probs_gt = logits_gt.permute(1, 0, 2).log_softmax(dim=-1)
            loss_ctc_gt = self.ctc(log_probs_gt, batch["targets"], input_lengths, batch["target_lengths"])

        total = (
            self.cls_weight * loss_cls
            + self.giou_weight * loss_giou
            + self.corner_weight * loss_corner
            + self.ctc_weight * loss_ctc
            + self.ctc_gt_weight * loss_ctc_gt
        )
        return {
            "loss": total,
            "cls": loss_cls.detach(),
            "giou": loss_giou.detach(),
            "corner": loss_corner.detach(),
            "ctc": loss_ctc.detach(),
            "ctc_gt": loss_ctc_gt.detach(),
        }
