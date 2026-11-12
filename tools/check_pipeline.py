"""Sanity-check the data path without training.

Draws the parsed GT quad on the letterboxed image and saves the rectified crop,
so annotation parsing / corner ordering / perspective warp can be verified by
eye.

    python tools/check_pipeline.py --out runs/sanity --n 8 [--split val]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpr.ccpd import build_loader  # noqa: E402
from lpr.geometry import warp_quads  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FONT_CANDIDATES = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                   r"C:\Windows\Fonts\simsun.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], size: int = 26) -> np.ndarray:
    """Draw a CJK-capable label (cv2's Hershey fonts are ASCII-only)."""
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = None
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            font = ImageFont.truetype(path, size)
            break
    if font is None:
        font = ImageFont.load_default()
    ImageDraw.Draw(pil).text(xy, text, font=font, fill=(0, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "runs" / "sanity"))
    args = ap.parse_args()

    data_root = ROOT / "data" / "CCPD2019"
    split = data_root / "splits" / f"{args.split}.txt"
    if not split.is_file():
        print(f"[FATAL] split file not found: {split}\n"
              f"        expected CCPD2019 under {data_root} (splits/{{train,val,test}}.txt "
              f"+ image subdirectories)")
        return 2
    loader = build_loader(data_root, split, batch_size=args.n, train=(args.split == "train"),
                          input_hw=(320, 512), workers=0, limit=args.n)
    batch = next(iter(loader))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = batch["image"]
    corners = batch["corners"]
    crop_hw = (32, 96)
    crops = warp_quads(image, corners * torch.tensor([512, 320]), crop_hw)

    def to_u8(t: torch.Tensor) -> np.ndarray:
        x = (t.clamp(-1, 1) + 1) * 127.5
        return cv2.cvtColor(x.permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

    tiles = []
    for i in range(args.n):
        vis = to_u8(image[i])
        pts = (corners[i].numpy() * np.array([512, 320])).astype(np.int32)
        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        crop = to_u8(crops[i])
        crop = cv2.resize(crop, (192, 64), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros_like(vis)
        canvas[8:8 + 64, 8:8 + 192] = crop
        canvas = draw_text(canvas, batch["texts"][i], (206, 30), 24)
        tiles.append(np.vstack([vis, canvas]))

    grid = np.vstack(tiles)
    out_path = out_dir / f"sanity_{args.split}.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"[ok] wrote {out_path}  ({args.n} samples)")
    print("[texts]", batch["texts"])
    print("[corner sample]", corners[0].tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
