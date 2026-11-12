"""Geometry helpers: quad ordering, point-in-quad test, GIoU and a differentiable
perspective warp used to crop the (predicted) plate region inside the network."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- numpy
def order_quad(quad: np.ndarray) -> np.ndarray:
    """Return the 4 points in a canonical ``[TL, TR, BR, BL]`` order.

    Points are sorted by polar angle around the centroid (clockwise for the
    image coordinate system, where y grows downwards), then rolled so that the
    top-left-most point comes first.
    """
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]

    # shoelace: positive for clockwise winding in a y-down coordinate frame
    area = float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
    if area < 0:
        q = q[::-1]
    start = int(np.argmin(q[:, 0] + q[:, 1]))
    return np.roll(q, -start, axis=0)


def quad_to_bbox(quad: np.ndarray) -> np.ndarray:
    """``(4, 2)`` -> ``(x1, y1, x2, y2)`` axis aligned box."""
    q = np.asarray(quad, dtype=np.float32)
    return np.array([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()], dtype=np.float32)


# -------------------------------------------------------------------------- torch
def quad_to_bbox_t(corners: torch.Tensor) -> torch.Tensor:
    """``(..., 4, 2)`` corners -> ``(..., 4)`` xyxy box."""
    x, y = corners[..., 0], corners[..., 1]
    return torch.stack([x.min(-1).values, y.min(-1).values, x.max(-1).values, y.max(-1).values], dim=-1)


def points_in_quad(points: torch.Tensor, quad: torch.Tensor) -> torch.Tensor:
    """Vectorised inside test for a convex quad (winding-order agnostic).

    Args:
        points: ``(N, 2)``
        quad: ``(4, 2)``
    Returns:
        ``(N,)`` bool tensor
    """
    crosses = []
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        crosses.append((b[0] - a[0]) * (points[:, 1] - a[1]) - (b[1] - a[1]) * (points[:, 0] - a[0]))
    crosses = torch.stack(crosses, dim=1)  # (N, 4)
    return (crosses >= 0).all(dim=1) | (crosses <= 0).all(dim=1)


def bbox_giou(box_a: torch.Tensor, box_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Generalised IoU for ``(N, 4)`` xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a.unbind(-1)
    bx1, by1, bx2, by2 = box_b.unbind(-1)

    inter_w = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(min=0)
    inter_h = (torch.min(ay2, by2) - torch.max(ay1, by1)).clamp(min=0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    union = area_a + area_b - inter + eps
    iou = inter / union

    cw = torch.max(ax2, bx2) - torch.min(ax1, bx1)
    ch = torch.max(ay2, by2) - torch.min(ay1, by1)
    hull = (cw.clamp(min=0) * ch.clamp(min=0)) + eps
    return iou - (hull - union) / hull


def homography_from_quads(dst_quad: torch.Tensor, src_quad: torch.Tensor) -> torch.Tensor:
    """Batched DLT: solve ``H`` such that ``src = H @ dst`` (both ``(B,4,2)``)."""
    b = dst_quad.shape[0]
    dtype, device = dst_quad.dtype, dst_quad.device
    a_mat = torch.zeros(b, 8, 8, dtype=dtype, device=device)
    b_vec = torch.zeros(b, 8, dtype=dtype, device=device)

    x, y = dst_quad[..., 0], dst_quad[..., 1]   # (B,4)
    src_x, src_y = src_quad[..., 0], src_quad[..., 1]

    a_mat[:, 0::2, 0] = x
    a_mat[:, 0::2, 1] = y
    a_mat[:, 0::2, 2] = 1.0
    a_mat[:, 0::2, 6] = -x * src_x
    a_mat[:, 0::2, 7] = -y * src_x
    b_vec[:, 0::2] = src_x

    a_mat[:, 1::2, 3] = x
    a_mat[:, 1::2, 4] = y
    a_mat[:, 1::2, 5] = 1.0
    a_mat[:, 1::2, 6] = -x * src_y
    a_mat[:, 1::2, 7] = -y * src_y
    b_vec[:, 1::2] = src_y

    # tiny ridge term keeps the system solvable for (nearly) degenerate quads
    eye = torch.eye(8, dtype=dtype, device=device).unsqueeze(0)
    h = torch.linalg.solve(a_mat + 1e-6 * eye, b_vec)  # (B,8)
    return torch.cat([h, torch.ones(b, 1, dtype=dtype, device=device)], dim=1).view(b, 3, 3)


def warp_quads(
    image: torch.Tensor,
    corners_px: torch.Tensor,
    out_hw: tuple[int, int],
    align_corners: bool = False,
) -> torch.Tensor:
    """Rectify the quads in ``corners_px`` into a fixed ``out_hw`` image.

    Args:
        image: ``(B, C, H, W)``
        corners_px: ``(B, 4, 2)`` plate corners in **pixel** coordinates, TL,TR,BR,BL
        out_hw: ``(out_h, out_w)`` output size
    Returns:
        ``(B, C, out_h, out_w)`` crops, differentiable w.r.t. ``corners_px``.
    """
    b, _, h, w = image.shape
    out_h, out_w = out_hw
    # Do the homography maths in fp32 even under fp16/bf16 autocast: bf16 linear
    # solves are slow and needlessly imprecise for a 8x8 system.
    corners_f = corners_px.float()

    dst = torch.tensor(
        [[0.0, 0.0], [out_w - 1.0, 0.0], [out_w - 1.0, out_h - 1.0], [0.0, out_h - 1.0]],
        dtype=torch.float32,
        device=image.device,
    ).unsqueeze(0).expand(b, -1, -1)

    mat = homography_from_quads(dst, corners_f)  # dst -> src

    ys, xs = torch.meshgrid(
        torch.arange(out_h, dtype=torch.float32, device=image.device),
        torch.arange(out_w, dtype=torch.float32, device=image.device),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=0).view(3, -1)          # (3, N)
    src = mat @ grid.unsqueeze(0)                                  # (B, 3, N)
    src = src[:, :2] / src[:, 2:3].clamp(min=1e-6)

    if align_corners:
        gx = src[:, 0] / (w - 1) * 2 - 1
        gy = src[:, 1] / (h - 1) * 2 - 1
    else:
        gx = (src[:, 0] + 0.5) / w * 2 - 1
        gy = (src[:, 1] + 0.5) / h * 2 - 1
    sample_grid = torch.stack([gx, gy], dim=-1).view(b, out_h, out_w, 2)
    return F.grid_sample(image, sample_grid.to(image.dtype), mode="bilinear",
                         padding_mode="border", align_corners=align_corners)
