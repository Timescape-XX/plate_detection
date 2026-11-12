"""Tiny end-to-end licence-plate network.

One forward pass on a raw image returns both the plate quad and the plate text:

    image (3,320,512)
      -> tiny residual backbone /4 /8 /16
      -> 3x3 conv neck at stride 8
      -> anchor-free head: objectness + 8 corner offsets  (grid 40x64)
      -> argmax cell -> plate quad (differentiable)
      -> perspective warp (grid_sample) -> 3x32x96 rectified crop
      -> small CRNN (CNN + BiLSTM) -> CTC logits (T=24, C=66)

Everything is differentiable, so detection and recognition are trained jointly
with one loss (Focal + L1 + GIoU + CTC).
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .charset import NUM_CLASSES
from .geometry import warp_quads


class ConvBNReLU(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int | None = None, groups: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """Two 3x3 convs + residual shortcut."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.short: nn.Module | None = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.short is None else self.short(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + idt)


class TinyBackbone(nn.Module):
    """~1.2M parameters, stride-8 features for the detector."""

    def __init__(self, width: float = 1.0):
        super().__init__()
        w = lambda c: max(8, int(round(c * width)))  # noqa: E731
        self.stem = ConvBNReLU(3, w(24), 3, 2)                     # /2

        self.stage1 = nn.Sequential(BasicBlock(w(24), w(24)))       # /2
        self.stage2 = nn.Sequential(BasicBlock(w(24), w(48), 2), BasicBlock(w(48), w(48)))  # /4
        self.stage3 = nn.Sequential(BasicBlock(w(48), w(96), 2), BasicBlock(w(96), w(96)))  # /8
        self.stage4 = nn.Sequential(BasicBlock(w(96), w(160), 2), BasicBlock(w(160), w(160)))  # /16

        self.out_channels = (w(96), w(160))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        c2 = self.stage2(x)   # /4
        c3 = self.stage3(c2)  # /8
        c4 = self.stage4(c3)  # /16
        return c2, c3, c4


class DetectionHead(nn.Module):
    """Anchor-free quad head: 1 objectness logit + 8 corner offsets per cell."""

    def __init__(self, cin: int, cmid: int = 128):
        super().__init__()
        self.block1 = ConvBNReLU(cin, cmid, 3, 1, 1)
        self.block2 = ConvBNReLU(cmid, cmid, 3, 1, 1)
        self.out = nn.Conv2d(cmid, 9, 1)
        nn.init.normal_(self.out.weight, std=0.001)
        nn.init.constant_(self.out.bias, 0.0)
        with torch.no_grad():
            self.out.bias[0] = -4.0  # focal-loss prior for objectness

    def forward(self, x: torch.Tensor):
        x = self.block2(self.block1(x))
        out = self.out(x)                 # (B, 9, gh, gw)
        obj, delta = out[:, :1], out[:, 1:]
        return obj, delta.view(delta.shape[0], 4, 2, delta.shape[2], delta.shape[3])


class RecognitionHead(nn.Module):
    """Small CRNN: 3x32x96 crop -> (T, C) CTC logits, T = crop_w // 4."""

    def __init__(self, crop_hw: Sequence[int] = (32, 96), num_classes: int = NUM_CLASSES,
                 rnn_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        crop_h, crop_w = int(crop_hw[0]), int(crop_hw[1])
        self.crop_hw = (crop_h, crop_w)

        self.cnn = nn.Sequential(
            ConvBNReLU(3, 32, 3, 1, 1), nn.MaxPool2d(2, 2),                       # 16 x w/2
            ConvBNReLU(32, 64, 3, 1, 1), nn.MaxPool2d(2, 2),                      # 8  x w/4
            ConvBNReLU(64, 128, 3, 1, 1), ConvBNReLU(128, 128, 3, 1, 1), nn.MaxPool2d((2, 1)),  # 4 x w/4
            ConvBNReLU(128, 192, 3, 1, 1), ConvBNReLU(192, 192, 3, 1, 1), nn.MaxPool2d((2, 1)),  # 2 x w/4
        )
        feat_h = crop_h // 16
        self.timesteps = crop_w // 4
        self.proj = nn.Linear(192 * feat_h, 256)
        self.rnn = nn.LSTM(256, rnn_hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(crop)                       # (B, 192, fh, T)
        b, c, fh, t = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(b, t, c * fh)
        feat = torch.relu(self.proj(feat))
        feat, _ = self.rnn(feat)
        return self.fc(self.drop(feat))             # (B, T, C)


class PlateNet(nn.Module):
    """End-to-end detector + recogniser."""

    def __init__(
        self,
        input_hw: Sequence[int] = (320, 512),
        stride: int = 8,
        crop_hw: Sequence[int] = (32, 96),
        width: float = 1.0,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.input_h, self.input_w = int(input_hw[0]), int(input_hw[1])
        self.stride = stride
        self.crop_hw = (int(crop_hw[0]), int(crop_hw[1]))
        self.grid_hw = (self.input_h // stride, self.input_w // stride)

        self.backbone = TinyBackbone(width)
        c3_ch, c4_ch = self.backbone.out_channels
        self.neck_c3 = nn.Conv2d(c3_ch, 96, 1)
        self.neck_c4 = nn.Conv2d(c4_ch, 96, 1)
        self.neck_fuse = ConvBNReLU(96, 128, 3, 1, 1)
        self.det_head = DetectionHead(128, 128)
        self.rec_head = RecognitionHead(self.crop_hw, num_classes)

        gh, gw = self.grid_hw
        ys = (torch.arange(gh, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(gw, dtype=torch.float32) + 0.5) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("cell_centers", torch.stack([xx, yy], dim=-1), persistent=False)

        # Prior for the corner branch: a plate-shaped box centred on the cell.
        # Keeps the initial warp non-degenerate (otherwise the homography solve
        # is singular and the CTC branch sees NaN crops at step 0).
        with torch.no_grad():
            dx = 0.105 * self.input_w / stride
            dy = 0.063 * self.input_h / stride
            self.det_head.out.bias[1:9] = torch.tensor(
                [-dx, -dy, dx, -dy, dx, dy, -dx, dy], dtype=self.det_head.out.bias.dtype
            )

    # ------------------------------------------------------------------ helpers
    def dense_corners(self, delta: torch.Tensor) -> torch.Tensor:
        """Per-cell corner predictions in pixel coordinates.

        Single source of truth shared by the loss and by :meth:`decode`, so the
        two can never disagree about the map layout.

        Args:
            delta: ``(B, 4, 2, gh, gw)`` offsets in units of ``stride``
        Returns:
            ``(B, gh*gw, 4, 2)`` absolute corner positions in input pixels.
        """
        b, _, _, gh, gw = delta.shape
        # (B,gh,gw,4,2) -> (B, N, 4, 2); contiguous() makes the merge explicit
        offset = delta.permute(0, 3, 4, 1, 2).contiguous().view(b, gh * gw, 4, 2)
        centers = self.cell_centers.view(1, gh * gw, 1, 2)
        return centers + offset * self.stride

    def decode(self, obj: torch.Tensor, delta: torch.Tensor):
        """Pick the best cell per image and decode its 4 corners (normalised)."""
        b, _, gh, gw = obj.shape
        scores = obj.sigmoid().reshape(b, -1)
        idx = scores.argmax(dim=1)                              # (B,)
        corners_px = self.dense_corners(delta)                  # (B,N,4,2)
        corners = corners_px.gather(
            1, idx.view(b, 1, 1, 1).expand(b, 1, 4, 2)
        ).squeeze(1)                                            # (B,4,2) pixels
        corners = torch.stack(
            [corners[..., 0] / self.input_w, corners[..., 1] / self.input_h], dim=-1
        ).clamp(0.0, 1.0)
        return corners, scores.gather(1, idx.view(-1, 1)).squeeze(1)

    # ------------------------------------------------------------------ forward
    def forward(self, image: torch.Tensor, return_crop: bool = False) -> Dict[str, torch.Tensor]:
        c2, c3, c4 = self.backbone(image)
        feat = self.neck_c3(c3) + F.interpolate(
            self.neck_c4(c4), size=c3.shape[-2:], mode="nearest"
        )
        feat = self.neck_fuse(feat)
        obj, delta = self.det_head(feat)                        # (B,1,gh,gw), (B,4,2,gh,gw)
        corners, scores = self.decode(obj, delta)

        corners_px = corners * torch.tensor(
            [self.input_w, self.input_h], dtype=corners.dtype, device=corners.device
        )
        crop = warp_quads(image, corners_px, self.crop_hw)
        logits = self.rec_head(crop)

        out = {"obj": obj, "delta": delta, "corners": corners, "scores": scores, "logits": logits}
        if return_crop:
            out["crop"] = crop
        return out

    # --------------------------------------------------------------------- misc
    @property
    def timesteps(self) -> int:
        return self.rec_head.timesteps

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def flops_estimate(self) -> float:
        """Rough forward FLOPs (multiply-adds) for the configured input size."""
        flops = 0.0
        h, w = self.input_h, self.input_w
        # backbone: dense convolutions dominate; approximates the residual stack
        flops += 9 * 3 * 24 * (h / 2) * (w / 2)
        for cin, cout, scale in [(24, 24, 2), (24, 48, 4), (48, 48, 4), (48, 96, 8),
                                 (96, 96, 8), (96, 160, 16), (160, 160, 16)]:
            flops += 2 * 9 * cin * cout * (h / scale) * (w / scale)
        return flops / 1e9
