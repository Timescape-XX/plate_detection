"""Timing breakdown of one training step (data wait vs. compute).

    python tools/profile_step.py [batch] [workers] [steps]
"""
from __future__ import annotations

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
    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    root = ROOT / "data" / "CCPD2019"
    loader = build_loader(root, root / "splits" / "train.txt", batch_size=bs, train=True,
                          input_hw=(320, 512), workers=workers, limit=1024)
    model = PlateNet((320, 512), 8, (32, 96)).cuda()
    crit = PlateLoss().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    it = iter(loader)
    t_wait = t_fwd = t_bwd = t_sync = 0.0
    for i in range(steps):
        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()
        img = batch["image"].cuda(non_blocking=True)
        bg = {"image": img, "corners": batch["corners"].cuda(non_blocking=True),
              "targets": batch["targets"].cuda(non_blocking=True),
              "target_lengths": batch["target_lengths"].cuda(non_blocking=True), "texts": batch["texts"]}
        t2 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(img)
            ls = crit(out, bg, model)
        t3 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        ls["loss"].backward()
        opt.step()
        t4 = time.perf_counter()
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        if i >= 2:
            t_wait += t1 - t0
            t_fwd += t3 - t2
            t_bwd += t4 - t3
            t_sync += t5 - t4
        print(f"  step {i}: wait {(t1-t0)*1000:6.1f}  fwd {(t3-t2)*1000:6.1f}  "
              f"bwd {(t4-t3)*1000:6.1f}  sync {(t5-t4)*1000:5.1f} ms", flush=True)
    n = max(steps - 2, 1)
    total = (t_wait + t_fwd + t_bwd + t_sync) / n
    print(f"bs={bs} workers={workers}: data {t_wait/n*1000:.0f} ms | fwd+loss {t_fwd/n*1000:.0f} ms | "
          f"bwd {t_bwd/n*1000:.0f} ms | sync {t_sync/n*1000:.0f} ms | total {total*1000:.0f} ms "
          f"({bs/total:.0f} img/s)")
    print(f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
