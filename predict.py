"""Run the trained model on images: draw the plate quad and the recognised text.

Paths may be written relative to the project directory (``plate_detection/``) no
matter where the shell is, because ``lpr.paths.resolve_path`` falls back to the
project root:

    python plate_detection/predict.py --weights runs/smoke_fix/checkpoints/best.pth --source test.jpg
    python plate_detection/predict.py --weights runs/smoke_fix/checkpoints/best.pth \
                                      --source my_folder/ --out runs/pred
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from lpr.ccpd import imread
from lpr.charset import ctc_greedy_decode
from lpr.model import PlateNet
from lpr.paths import resolve_path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# CJK-capable TrueType fonts. The plate text starts with a Chinese province
# character, and OpenCV's Hershey fonts are ASCII-only, so PIL + a system font is
# required. Windows paths come first only because that is where the project was
# developed; the macOS/Linux entries make the same code work elsewhere.
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    # macOS
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

_warned_font = False


def find_font(explicit: str | Path | None = None) -> Path | None:
    """First existing CJK font, honouring an explicit ``--font`` override."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        print(f"[warn] font not found: {p}", file=sys.stderr)
        return None
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return Path(path)
    return None


def load_model(weights: str | Path, device: torch.device) -> PlateNet:
    ckpt = torch.load(str(weights), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mc = cfg.get("model", {})
    model = PlateNet(
        input_hw=(mc.get("input_h", 320), mc.get("input_w", 512)),
        stride=mc.get("stride", 8),
        crop_hw=(mc.get("crop_h", 32), mc.get("crop_w", 96)),
        width=mc.get("width", 1.0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def preprocess(img: np.ndarray, input_h: int, input_w: int) -> tuple[torch.Tensor, float, int, int]:
    h, w = img.shape[:2]
    scale = min(input_w / w, input_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    pad_x, pad_y = (input_w - nw) // 2, (input_h - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(img, (nw, nh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    return torch.from_numpy(rgb.transpose(2, 0, 1))[None], scale, pad_x, pad_y


@torch.no_grad()
def predict_image(model: PlateNet, img: np.ndarray, device: torch.device) -> dict:
    tensor, scale, pad_x, pad_y = preprocess(img, model.input_h, model.input_w)
    out = model(tensor.to(device))
    corners = out["corners"][0].cpu().numpy()
    # back to original image coordinates
    corners = (corners * np.array([model.input_w, model.input_h]) - np.array([pad_x, pad_y])) / scale
    text = ctc_greedy_decode(out["logits"])[0]
    return {"corners": corners, "text": text, "score": float(out["scores"][0])}


def draw(img: np.ndarray, corners: np.ndarray, text: str,
         font_path: str | Path | None = None) -> np.ndarray:
    """Draw the plate quad plus a filled label with the recognised text."""
    vis = img.copy()
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    if not text:                       # nothing recognised: do not draw an empty bar
        return vis

    h, w = vis.shape[:2]
    size = max(16, int(round(min(w, h) * 0.04)))     # 720x1160 -> 28 px
    x, y = int(corners[:, 0].min()), int(corners[:, 1].min())

    font_file = find_font(font_path)
    if font_file is None:
        global _warned_font
        if not _warned_font:
            _warned_font = True
            print("[warn] no CJK font found, falling back to OpenCV text "
                  "(Chinese province glyph will be missing); pass --font /path/to/font.ttc",
                  file=sys.stderr)
        cv2.putText(vis, text, (x, max(size, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    size / 32, (0, 255, 0), 2, cv2.LINE_AA)
        return vis

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_file), size)
    pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    painter = ImageDraw.Draw(pil)
    l, t, r, b = painter.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    pad = max(3, size // 6)
    bx0 = min(max(0, x), max(0, w - tw - 2 * pad))
    by0 = y - th - 2 * pad
    if by0 < 0:                        # plate near the top edge: label goes inside
        by0 = min(y + 2, max(0, h - th - 2 * pad))
    painter.rectangle([bx0, by0, bx0 + tw + 2 * pad, by0 + th + 2 * pad], fill=(0, 180, 0))
    painter.text((bx0 + pad - l, by0 + pad - t), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="checkpoint, e.g. runs/smoke_fix/checkpoints/best.pth")
    ap.add_argument("--source", required=True, help="image file or directory")
    ap.add_argument("--out", default="runs/pred", help="output directory (default: runs/pred)")
    ap.add_argument("--font", default=None, help="TrueType font for the label (default: auto-detect)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    weights = resolve_path(args.weights)
    if not weights.is_file():
        print(f"[FATAL] checkpoint not found: {args.weights} (tried {weights})", file=sys.stderr)
        return 2
    model = load_model(weights, device)

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = resolve_path(args.source)
    if not src.exists():
        print(f"[FATAL] source not found: {args.source} (tried {src})", file=sys.stderr)
        return 2
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    for path in files:
        img = imread(path)
        if img is None:
            print(f"[skip] {path}")
            continue
        res = predict_image(model, img, device)
        vis = draw(img, res["corners"], res["text"], args.font)
        dst = out_dir / f"{path.stem}_pred.jpg"
        cv2.imwrite(str(dst), vis)
        print(f"{path.name}: {res['text']}  score={res['score']:.3f}  "
              f"corners={np.round(res['corners'], 1).tolist()}  -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
