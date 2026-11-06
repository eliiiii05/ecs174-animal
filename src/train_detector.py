"""Training process 2: Faster R-CNN (ResNet-18 + FPN) detector.

    python -m src.train_detector --config configs/det_D1_agnostic.yaml
    python -m src.train_detector --config configs/det_D1_agnostic.yaml --resume

"""
import argparse
import csv
import math
import time

import torch
from torch.utils.data import DataLoader

from src.data.datasets import DetectionDataset, detection_collate
from src.det_eval import coco_eval, load_gt, run_detector
from src.engine import get_device
from src.models.detector import build_detector
from src.utils import REPO_ROOT, append_results, load_config, resolve, set_seed


def lr_at(cfg, it, epoch):
    """Linear warmup over warmup_iters, then x gamma after each milestone epoch."""
    if it < cfg["warmup_iters"]:
        alpha = it / cfg["warmup_iters"]
        warm = cfg["warmup_factor"] * (1 - alpha) + alpha
    else:
        warm = 1.0
    decay = cfg["gamma"] ** sum(epoch >= m for m in cfg["milestones"])
    return cfg["lr"] * warm * decay


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device()
    amp = cfg["amp"] and device.type == "cuda"
    run_dir = REPO_ROOT / "runs" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    data = resolve(cfg["data_root"])
    agnostic = cfg["agnostic"]

    train_ds = DetectionDataset(data / "annotations" / "instances_train.json", data / "images",
                                agnostic=agnostic, train=True)
    val_ds = DetectionDataset(data / "annotations" / "instances_val.json", data / "images",
                              agnostic=agnostic, train=False)
    val_gt = load_gt(data / "annotations" / "instances_val.json", agnostic=agnostic)
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                        collate_fn=detection_collate, persistent_workers=cfg["num_workers"] > 0)
    print(f"train images: {len(train_ds)}, val images: {len(val_ds)}")

    num_classes = 2 if agnostic else 11
    model = build_detector(num_classes, cfg["min_size"], cfg["max_size"], cfg["trainable_layers"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, it, best_ap50, train_seconds = 0, 0, -1.0, 0.0
    if args.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, it = ck["epoch"] + 1, ck["iter"]
        best_ap50, train_seconds = ck["best_ap50"], ck["train_seconds"]
        print(f"Resumed from epoch {start_epoch}")

    log_path = run_dir / "log.csv"
    if start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_AP50", "val_AP", "epoch_minutes"])

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t0, total, n, skipped = time.time(), 0.0, 0, 0
        for step, (imgs, targets, _) in enumerate(loader):
            for g in opt.param_groups:
                g["lr"] = lr_at(cfg, it, epoch)
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            if not math.isfinite(loss.item()):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                it += 1
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            n += 1
            it += 1
            if step % 200 == 0:
                parts = "  ".join(f"{k.replace('loss_', '')} {v.item():.3f}" for k, v in loss_dict.items())
                print(f"epoch {epoch + 1} step {step}/{len(loader)}  loss {loss.item():.3f}  ({parts})  "
                      f"lr {opt.param_groups[0]['lr']:.5f}")
        minutes = (time.time() - t0) / 60
        train_seconds += minutes * 60
        if skipped:
            print(f"WARNING: skipped {skipped} steps with non-finite loss")

        ap50 = ap = float("nan")
        if (epoch + 1) % cfg["eval_every"] == 0 or epoch + 1 == cfg["epochs"]:
            res = run_detector(model, val_ds, device, batch_size=cfg["batch_size"],
                               num_workers=cfg["num_workers"], amp=amp)
            stats, _ = coco_eval(val_gt, res)
            ap50, ap = stats["AP50"], stats["AP"]
            if ap50 > best_ap50:
                best_ap50 = ap50
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                            "val_stats": stats}, run_dir / "best.pt")
        print(f"epoch {epoch + 1}/{cfg['epochs']}  loss {total / max(n, 1):.4f}  "
              f"val AP50 {ap50:.4f}  AP {ap:.4f}  ({minutes:.1f} min)")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, round(total / max(n, 1), 5), round(ap50, 5), round(ap, 5),
                                    round(minutes, 1)])
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "iter": it, "best_ap50": best_ap50, "train_seconds": train_seconds,
                    "config": cfg}, run_dir / "last.pt")

    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    vs = best["val_stats"]
    append_results(cfg["run_name"], "fasterrcnn_r18_fpn", cfg["_config_file"],
                   "val_agnostic" if agnostic else "val",
                   {"AP50": vs["AP50"], "AP": vs["AP"], "best_epoch": best["epoch"] + 1,
                    "train_minutes": train_seconds / 60})
    print(f"Done. Best val AP50 {vs['AP50']:.4f}; total training time {train_seconds / 3600:.2f} h")


if __name__ == "__main__":
    main()
