"""Throughput / memory benchmark: measures data loading, forward-backward time
and peak GPU memory for a range of batch sizes.

    python tools/bench.py --batches 8 16 32 64 --steps 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpr.ccpd import build_loader  # noqa: E402
from lpr.loss import PlateLoss  # noqa: E402
from lpr.model import PlateNet  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--input-hw", type=int, nargs=2, default=[320, 512])
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-only", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    data_root = ROOT / "data" / "CCPD2019"
    split = data_root / "splits" / "train.txt"

    # ---- data pipeline -----------------------------------------------------
    t0 = time.perf_counter()
    loader = build_loader(data_root, split, batch_size=16, train=True,
                          input_hw=tuple(args.input_hw), workers=args.workers, limit=512)
    t_data = 0.0
    n_img = 0
    for batch in loader:
        t_data += 0  # timer below
    t_iter = 0.0
    t_mark = time.perf_counter()
    for batch in loader:
        n_img += batch["image"].shape[0]
    t_iter = time.perf_counter() - t_mark
    first_batch = batch
    print(f"[data] workers={args.workers} images={n_img} "
          f"throughput={n_img / max(t_iter, 1e-6):.0f} img/s "
          f"({t_iter / max(len(loader), 1) * 1000:.0f} ms/batch16)")
    if args.data_only:
        return 0

    # ---- model ------------------------------------------------------------
    results = []
    for bs in args.batches:
        model = PlateNet(tuple(args.input_hw), 8, (32, 96)).to(device)
        crit = PlateLoss().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        def make_batch(b: int):
            img = first_batch["image"][:1].repeat(b, 1, 1, 1).to(device)
            lab_len = int(first_batch["target_lengths"][0])
            labels = first_batch["targets"][:lab_len]
            return {
                "image": img,
                "corners": first_batch["corners"][:1].repeat(b, 1, 1).to(device),
                "targets": labels.repeat(b).to(device),
                "target_lengths": torch.full((b,), lab_len, dtype=torch.long, device=device),
                "texts": [""] * b,
            }

        bg = make_batch(bs)
        # bf16 inputs are only used for autocast; keep masters in fp32
        use_amp = args.amp != "off" and device.type == "cuda"
        dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        scaler = torch.amp.GradScaler("cuda") if (use_amp and args.amp == "fp16") else None

        def one_step():
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                out = model(bg["image"])
                ls = crit(out, bg, model)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(ls["loss"]).backward()
                scaler.step(opt)
                scaler.update()
            else:
                ls["loss"].backward()
                opt.step()
            return float(ls["loss"].detach())

        for _ in range(3):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(args.steps):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / args.steps
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
        reserved = torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0.0
        results.append({"batch": bs, "ms_per_step": round(dt * 1000, 1),
                        "img_per_s": round(bs / dt, 1), "peak_mb": round(peak, 1),
                        "reserved_mb": round(reserved, 1)})
        print(f"[train] bs={bs:>3} {dt*1000:7.1f} ms/step  {bs/dt:7.1f} img/s  "
              f"peak={peak:7.1f} MB  reserved={reserved:7.1f} MB")
        del model, crit, opt, bg
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(json.dumps(results, indent=2))
    out = ROOT / "runs" / "bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
