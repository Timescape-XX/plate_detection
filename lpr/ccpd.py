"""CCPD2019 dataset: filename annotation parsing, letterbox preprocessing,
photo-metric / geometric augmentation and a CTC-friendly collate function."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .charset import decode_label_indices
from .geometry import order_quad

cv2.setNumThreads(0)


# ---------------------------------------------------------------------------- io
def imread(path: str | Path) -> np.ndarray | None:
    """Unicode-safe BGR read."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def parse_ccpd_name(name: str) -> Dict:
    """Parse a CCPD filename into ``{corners, text, bbox, tilt, area}``.

    ``0022-0_1-361&474_433&500-432&500_360&499_361&474_433&475-0_0_20_...-66-5.jpg``
    fields: area-tilt-bbox-vertices-labels-brightness-blur
    """
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"not a CCPD annotation filename: {name}")

    def to_points(field: str) -> np.ndarray:
        return np.array([[float(v) for v in p.split("&")] for p in field.split("_")], dtype=np.float32)

    vertices = to_points(parts[3])
    if vertices.shape != (4, 2):
        raise ValueError(f"expected 4 vertices in {name}")
    labels = [int(i) for i in parts[4].split("_")]
    bbox = to_points(parts[2])
    return {
        "corners": order_quad(vertices),
        "text": decode_label_indices(labels),
        "labels": labels,
        "bbox": bbox,
        "area": int(parts[0].split("_")[0]),
        "tilt": [int(v) for v in parts[1].split("_")],
    }


def resolve_split(root: Path, split_file: str | Path) -> List[str]:
    """Read a split file and return paths that exist (handles the nested
    ``ccpd_base/ccpd_base/`` layout of this dataset copy)."""
    lines = [
        ln.strip()
        for ln in Path(split_file).read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    out: List[str] = []
    for rel in lines:
        p = root / rel
        if not p.is_file():
            parts = Path(rel).parts
            if len(parts) >= 2:
                alt = root / parts[0] / parts[0] / Path(*parts[1:])
                if alt.is_file():
                    p = alt
                else:
                    continue
            else:
                continue
        out.append(str(p))
    return out


# ----------------------------------------------------------------------- dataset
class CCPDDataset(Dataset):
    """Yields letterboxed images, normalised plate corners and CTC targets."""

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path,
        input_hw: Sequence[int] = (320, 512),
        train: bool = True,
        limit: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
        max_rotate: float = 6.0,
        scale_ratio: tuple[float, float] = (0.85, 1.15),
        translate: float = 0.02,
        perspective_prob: float = 0.3,
        perspective_jitter: float = 0.02,
        color_prob: float = 0.8,
        blur_prob: float = 0.15,
        skip_broken: bool = True,
    ) -> None:
        self.root = Path(root)
        self.input_h, self.input_w = int(input_hw[0]), int(input_hw[1])
        self.train = train
        self.max_rotate = max_rotate
        self.scale_ratio = scale_ratio
        self.translate = translate
        self.perspective_prob = perspective_prob
        self.perspective_jitter = perspective_jitter
        self.color_prob = color_prob
        self.blur_prob = blur_prob
        self.skip_broken = skip_broken

        files = resolve_split(self.root, split_file)
        if shuffle:
            random.Random(seed).shuffle(files)
        if limit is not None:
            files = files[:limit]
        self.files = files
        if not self.files:
            raise RuntimeError(f"no usable images found for split {split_file} under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    # -------------------------------------------------------------- transforms
    def _load(self, path: str):
        img = imread(path)
        ann = parse_ccpd_name(Path(path).name)
        return img, ann

    def _letterbox(self, img: np.ndarray, corners: np.ndarray):
        h, w = img.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        pad_x = (self.input_w - nw) // 2
        pad_y = (self.input_h - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        corners = corners * scale + np.array([pad_x, pad_y], dtype=np.float32)
        return canvas, corners

    def _augment(self, img: np.ndarray, corners: np.ndarray):
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        angle = random.uniform(-self.max_rotate, self.max_rotate)
        scale = random.uniform(*self.scale_ratio)
        m2 = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        m2[0, 2] += random.uniform(-self.translate, self.translate) * w
        m2[1, 2] += random.uniform(-self.translate, self.translate) * h
        mat = np.vstack([m2, [0, 0, 1]]).astype(np.float32)

        if random.random() < self.perspective_prob:
            jx, jy = self.perspective_jitter * w, self.perspective_jitter * h
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [random.uniform(0, jx), random.uniform(0, jy)],
                [w - random.uniform(0, jx), random.uniform(0, jy)],
                [w - random.uniform(0, jx), h - random.uniform(0, jy)],
                [random.uniform(0, jx), h - random.uniform(0, jy)],
            ])
            mat = cv2.getPerspectiveTransform(src, dst).astype(np.float32) @ mat

        img = cv2.warpPerspective(img, mat, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        pts = cv2.perspectiveTransform(corners.reshape(1, 4, 2).astype(np.float32), mat)[0]
        return img, pts.astype(np.float32)

    def _photometric(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.color_prob:
            out = img.astype(np.float32)
            out *= random.uniform(0.75, 1.25)              # brightness / contrast
            out += random.uniform(-18, 18)
            if random.random() < 0.4:                      # gamma
                out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, random.uniform(0.75, 1.4))
            img = np.clip(out, 0, 255).astype(np.uint8)
        if random.random() < self.blur_prob:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        return img

    # ------------------------------------------------------------------ getitem
    def __getitem__(self, idx: int) -> Dict:
        path = self.files[idx]
        img, ann = self._load(path)
        if img is None:
            if not self.skip_broken:
                raise RuntimeError(f"cannot decode {path}")
            return self.__getitem__((idx + 1) % len(self))

        corners = ann["corners"].copy()
        canvas, corners = self._letterbox(img, corners)
        if self.train:
            canvas, corners = self._augment(canvas, corners)
            canvas = self._photometric(canvas)

        corners[:, 0] = np.clip(corners[:, 0], 0, self.input_w - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, self.input_h - 1)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        image = torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous()

        norm_corners = corners / np.array([self.input_w, self.input_h], dtype=np.float32)
        return {
            "image": image,
            "corners": torch.from_numpy(norm_corners),
            "target": torch.tensor(ann["labels"], dtype=torch.long),
            "text": ann["text"],
            "path": path,
        }


def ctc_collate(batch: List[Dict]) -> Dict:
    """Stack images/corners, flatten variable-length CTC targets."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    corners = torch.stack([b["corners"] for b in batch], dim=0)
    targets = torch.cat([b["target"] for b in batch], dim=0)
    lengths = torch.tensor([b["target"].numel() for b in batch], dtype=torch.long)
    return {
        "image": images,
        "corners": corners,
        "targets": targets,
        "target_lengths": lengths,
        "texts": [b["text"] for b in batch],
        "paths": [b["path"] for b in batch],
    }


def build_loader(
    root: str | Path,
    split_file: str | Path,
    batch_size: int,
    train: bool,
    input_hw: Sequence[int] = (320, 512),
    workers: int = 4,
    limit: int | None = None,
    seed: int = 0,
    **kwargs,
) -> DataLoader:
    ds = CCPDDataset(root, split_file, input_hw=input_hw, train=train, limit=limit, **kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        drop_last=train and len(ds) > batch_size,
        collate_fn=ctc_collate,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
