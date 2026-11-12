"""Training entry point for the tiny end-to-end plate network.

    python plate_detection/train.py --config configs/default.yaml
    python plate_detection/train.py --max-steps 50 --batch-size 8 --output-dir runs/smoke

Relative paths are resolved against the current directory first and against this
project directory second, so ``runs/...`` / ``configs/...`` work from either the
repository root or from inside ``plate_detection/`` (see ``lpr/paths.py``).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml

from lpr.ccpd import build_loader, build_train_loader
from lpr.evaluate import run_eval
from lpr.loss import PlateLoss
from lpr.model import PlateNet
from lpr.paths import PROJECT_ROOT, resolve_path


# ----------------------------------------------------------------------- helpers
def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    cfg["data"]["root"] = str((PROJECT_ROOT / cfg["data"]["root"]).resolve()
                              if not Path(cfg["data"]["root"]).is_absolute() else cfg["data"]["root"])
    for key in ("train_split", "val_split"):
        p = Path(cfg["data"][key])
        cfg["data"][key] = str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())
    cfg["output_dir"] = str((PROJECT_ROOT / cfg["output_dir"]).resolve()
                            if not Path(cfg["output_dir"]).is_absolute() else cfg["output_dir"])
    return cfg


def lr_at(step: int, total: int, base_lr: float, warmup: int, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cos = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return base_lr * (min_ratio + (1 - min_ratio) * cos)


def make_amp(device: torch.device, mode: str):
    """Return (autocast_ctx_factory, scaler)."""
    if mode == "off" or device.type != "cuda":
        from contextlib import nullcontext
        return (lambda: nullcontext()), None
    use_bf16 = mode == "bf16" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None if use_bf16 else torch.amp.GradScaler("cuda")
    return (lambda: torch.amp.autocast("cuda", dtype=dtype)), scaler


def format_metrics(m: dict) -> str:
    keys = ["iou", "det_rate", "plate_acc", "plate_acc_when_detected", "cer"]
    return " ".join(f"{k}={m[k]:.4f}" for k in keys if k in m)


# -------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke test)")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="stop after N epochs while keeping the full cosine horizon "
                         "(used for the equal-budget α=0 control run)")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--val-batches", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--tag", default=None, help="suffix for the run directory")
    ap.add_argument("--alpha", type=float, default=None,
                    help="override train.sampling.province_alpha (0=natural prior, 1=uniform)")
    ap.add_argument("--no-val", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc, mc, dc = cfg["train"], cfg["model"], cfg["data"]

    required = [dc["root"], dc["train_split"]] + ([] if args.no_val else [dc["val_split"]])
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        print("[FATAL] dataset not found: " + ", ".join(missing)
              + f"\n        expected under {PROJECT_ROOT / 'data' / 'CCPD2019'}"
              + " (splits/{train,val,test}.txt + image subdirectories)", flush=True)
        return 2

    batch_size = args.batch_size or tc["batch_size"]
    lr = args.lr or tc["lr"]
    epochs = args.epochs or tc["epochs"]
    workers = tc["workers"] if args.workers is None else args.workers
    val_every = args.val_every or tc["val_every"]
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else tc["warmup_steps"]
    log_every = args.log_every or tc["log_every"]
    val_batches = args.val_batches if args.val_batches is not None else tc.get("val_batches")
    max_steps = args.max_steps
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.tag:
        cfg["output_dir"] = f"{cfg['output_dir']}_{args.tag}"
    if args.output_dir:
        # resolve against the project root so `--output-dir runs/smoke` lands next
        # to the checkpoints no matter which directory the command was run from
        cfg["output_dir"] = str(resolve_path(args.output_dir))
    out_dir = Path(cfg["output_dir"])
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    torch.manual_seed(tc["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(tc["seed"])
        torch.backends.cudnn.benchmark = True

    input_hw = (mc["input_h"], mc["input_w"])
    sampling = dict(tc.get("sampling") or {})
    if args.alpha is not None:
        sampling["province_alpha"] = args.alpha
    train_loader, plan = build_train_loader(
        dc["root"], dc["train_split"], batch_size=batch_size,
        input_hw=input_hw, workers=workers, limit=args.train_limit,
        sampling=sampling or None,
        max_rotate=dc["aug"]["max_rotate"], scale_ratio=tuple(dc["aug"]["scale_ratio"]),
        translate=dc["aug"]["translate"], perspective_prob=dc["aug"]["perspective_prob"],
        perspective_jitter=dc["aug"]["perspective_jitter"], color_prob=dc["aug"]["color_prob"],
        blur_prob=dc["aug"]["blur_prob"],
    )
    val_loader = None
    if not args.no_val:
        val_loader = build_loader(
            dc["root"], dc["val_split"], batch_size=batch_size, train=False,
            input_hw=input_hw, workers=max(2, workers // 2),
            limit=args.val_limit if args.val_limit is not None else dc.get("val_limit"),
        )

    model = PlateNet(
        input_hw=input_hw, stride=mc["stride"], crop_hw=(mc["crop_h"], mc["crop_w"]),
        width=mc["width"],
    ).to(device)
    criterion = PlateLoss(**cfg["loss"]).to(device)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tc["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.999),
    )
    amp_ctx, scaler = make_amp(device, tc.get("amp", "bf16"))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max_steps or steps_per_epoch * epochs
    print(f"[setup] device={device} params={model.num_parameters()/1e6:.2f}M "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset) if val_loader else 0} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} bs={batch_size} "
          f"input={input_hw} amp={tc.get('amp')}", flush=True)
    if plan is not None:
        print(plan.summary(), flush=True)
        (out_dir / "sampling_plan.json").write_text(json.dumps({
            "alpha": plan.alpha,
            "province_natural": plan.natural,
            "province_effective": plan.effective,
            "folder_effective": plan.folder_effective,
            "config": sampling,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print("[sampler] uniform shuffling over the natural prior", flush=True)

    start_step = 0
    best_acc = -1.0
    best_macro = -1.0
    val_history: list[dict] = []
    if args.resume:
        resume_path = resolve_path(args.resume)
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        best_acc = ckpt.get("metrics", {}).get("plate_acc", -1.0)
        print(f"[resume] {resume_path} @ step {start_step}", flush=True)

    log_path = out_dir / "train_log.csv"
    log_file = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if log_path.stat().st_size == 0:
        writer.writerow(["step", "epoch", "lr", "loss", "cls", "giou", "corner", "ctc", "ctc_gt",
                         "sec_per_step", "val_iou", "val_det_rate", "val_plate_acc", "val_cer",
                         "val_wan_fp", "val_macro_head"])

    model.train()
    step = start_step
    t_start = time.time()
    t_last = t_start
    running = {k: 0.0 for k in ("loss", "cls", "giou", "corner", "ctc", "ctc_gt")}
    running_n = 0
    epoch = 0

    def save(name: str, metrics: dict | None = None) -> None:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": cfg,
            "metrics": metrics or {},
        }, out_dir / "checkpoints" / name)

    try:
        while step < total_steps:
            epoch += 1
            for batch in train_loader:
                cur_lr = lr_at(step, total_steps, lr, warmup_steps)
                for g in optimizer.param_groups:
                    g["lr"] = cur_lr

                image = batch["image"].to(device, non_blocking=True)
                batch_gpu = {
                    "image": image,
                    "corners": batch["corners"].to(device, non_blocking=True),
                    "targets": batch["targets"].to(device, non_blocking=True),
                    "target_lengths": batch["target_lengths"].to(device, non_blocking=True),
                    "texts": batch["texts"],
                }

                with amp_ctx():
                    outputs = model(image)
                    losses = criterion(outputs, batch_gpu, model)
                total_loss = losses["loss"]

                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                    optimizer.step()

                step += 1
                for k in running:
                    running[k] += float(losses[k].detach())
                running_n += 1

                if step % log_every == 0 or step == total_steps:
                    now = time.time()
                    sps = (now - t_last) / max(running_n, 1)
                    t_last = now
                    msg = (f"[train] step {step}/{total_steps} ep {epoch} lr {cur_lr:.2e} "
                           + " ".join(f"{k} {running[k]/running_n:.4f}" for k in running)
                           + f" | {sps:.3f}s/step")
                    print(msg, flush=True)
                    writer.writerow([step, epoch, f"{cur_lr:.6f}",
                                     *[f"{running[k]/running_n:.5f}" for k in running],
                                     f"{sps:.4f}"])
                    log_file.flush()
                    running = {k: 0.0 for k in running}
                    running_n = 0

                if val_loader is not None and (step % val_every == 0 or step == total_steps):
                    metrics = run_eval(model, val_loader, device, max_batches=val_batches)
                    print(f"[val  ] step {step} " + format_metrics(metrics)
                          + f" wan_fp={metrics.get('wan_fp_rate', 0):.3f}"
                          + f" macro_head={metrics.get('macro_acc_nonwan_head', 0):.3f}", flush=True)
                    print(f"        examples: {[(e['pred'], e['gt']) for e in metrics['examples'][:4]]}", flush=True)
                    writer.writerow([step, epoch, f"{cur_lr:.6f}", "", "", "", "", "", "", "",
                                     f"{metrics['iou']:.5f}", f"{metrics['det_rate']:.5f}",
                                     f"{metrics['plate_acc']:.5f}", f"{metrics['cer']:.5f}",
                                     f"{metrics.get('wan_fp_rate', 0):.5f}",
                                     f"{metrics.get('macro_acc_nonwan_head', 0):.5f}"])
                    log_file.flush()
                    acc = metrics["plate_acc"]
                    if acc > best_acc:
                        best_acc = acc
                        save("best.pth", metrics)
                        print(f"[ckpt ] best.pth (plate_acc={acc:.4f})", flush=True)
                    macro = metrics.get("macro_acc_nonwan_head", 0.0)
                    if macro > best_macro:
                        best_macro = macro
                        save("best_nonwan.pth", metrics)
                        print(f"[ckpt ] best_nonwan.pth (macro_head={macro:.4f})", flush=True)
                    model.train()
                    save("last.pth", metrics)
                    val_history.append({
                        "step": step, "epoch": epoch,
                        "iou": metrics["iou"], "det_rate": metrics["det_rate"],
                        "plate_acc": metrics["plate_acc"], "cer": metrics["cer"],
                        "wan_fp_rate": metrics.get("wan_fp_rate", 0.0),
                        "macro_acc_nonwan_head": metrics.get("macro_acc_nonwan_head", 0.0),
                        "by_province": metrics.get("by_province", {}),
                    })
                elif step == total_steps:
                    save("last.pth")

                if step >= total_steps:
                    break
            # one full pass over the (re-weighted) pool: keep it for the curve plot
            if step < total_steps and (args.max_epochs is None or epoch < args.max_epochs):
                save(f"epoch_{epoch:02d}.pth")
                print(f"[epoch] {epoch} done @ step {step}", flush=True)
            if args.max_epochs is not None and epoch >= args.max_epochs:
                print(f"[stop] reached --max-epochs {args.max_epochs} @ step {step}, "
                      f"saving last.pth", flush=True)
                save("last.pth")
                break
    except KeyboardInterrupt:
        print("[stop] interrupted, saving last.pth", flush=True)
        save("last.pth")
    finally:
        log_file.close()

    elapsed = time.time() - t_start
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0
    summary = {
        "steps": step,
        "elapsed_sec": round(elapsed, 1),
        "sec_per_step": round(elapsed / max(step - start_step, 1), 4),
        "batch_size": batch_size,
        "params_m": round(model.num_parameters() / 1e6, 3),
        "peak_allocated_mb": round(peak_mb, 1),
        "peak_reserved_mb": round(peak_reserved, 1),
        "best_plate_acc": best_acc,
        "best_macro_nonwan_head": best_macro,
        "sampling_alpha": sampling.get("province_alpha"),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if val_history:
        (out_dir / "val_history.json").write_text(
            json.dumps(val_history, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[done] " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    raise SystemExit(main())
