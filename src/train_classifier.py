"""Training process 1: fine-tune the ResNet-18 classifier on GT box crops.

    python -m src.train_classifier --config configs/cls_C1_finetune_aug.yaml
    python -m src.train_classifier --config configs/cls_C1_finetune_aug.yaml --resume
    python -m src.train_classifier --config configs/cls_C1_finetune_aug.yaml \\
        --set data_root=data_debug epochs=1 run_name=debug_C1      # smoke test

Writes runs/<run_name>/{best.pt,last.pt,log.csv} and appends the best
validation scores to results/results.csv.
"""
import argparse
import csv
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import CropDataset, make_balanced_sampler
from src.data.transforms import build_crop_transform
from src.engine import get_device, predict_classifier
from src.metrics import classification_metrics
from src.models.classifier import build_classifier, freeze_batchnorm
from src.utils import REPO_ROOT, append_results, load_config, resolve, set_seed


def build_optimizer(model, cfg):
    head = [p for n, p in model.named_parameters() if n.startswith("fc.") and p.requires_grad]
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.") and p.requires_grad]
    groups = [{"params": head, "lr": cfg["lr_head"]}]
    if backbone:
        groups.append({"params": backbone, "lr": cfg["lr_backbone"]})
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def warmup_cosine(total_steps, warmup_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override config values, e.g. epochs=1")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<run_name>/last.pt")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device()
    amp = cfg["amp"] and device.type == "cuda"
    run_dir = REPO_ROOT / "runs" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    data = resolve(cfg["data_root"])

    train_ds = CropDataset(data / "crops" / "train.csv", data, build_crop_transform(cfg["preproc"], train=True))
    val_ds = CropDataset(data / "crops" / "val.csv", data, build_crop_transform("A", train=False))
    sampler = make_balanced_sampler(train_ds.labels) if cfg["balanced_sampler"] else None
    loader_kw = dict(num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=cfg["num_workers"] > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=sampler is None,
                              sampler=sampler, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False, **loader_kw)
    print(f"train crops: {len(train_ds)}, val crops: {len(val_ds)}")

    model = build_classifier(pretrained=cfg["pretrained"], freeze_backbone=cfg["freeze_backbone"]).to(device)
    opt = build_optimizer(model, cfg)
    steps = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(cfg["epochs"] * steps, cfg["warmup_epochs"] * steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    start_epoch, best_f1, train_seconds = 0, -1.0, 0.0
    if args.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_f1, train_seconds = ck["epoch"] + 1, ck["best_f1"], ck["train_seconds"]
        print(f"Resumed from epoch {start_epoch}")

    log_path = run_dir / "log.csv"
    log_fields = ["epoch", "train_loss", "val_accuracy", "val_macro_f1", "lr_head", "epoch_seconds"]
    if start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(log_fields)

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        if cfg["freeze_backbone"]:
            freeze_batchnorm(model)
        t0, total_loss, n = time.time(), 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
        epoch_seconds = time.time() - t0
        train_seconds += epoch_seconds

        y_true, y_pred, _, _ = predict_classifier(model, val_loader, device, amp)
        m = classification_metrics(y_true, y_pred)
        print(f"epoch {epoch + 1:2d}/{cfg['epochs']}  loss {total_loss / max(n, 1):.4f}  "
              f"val acc {m['accuracy']:.4f}  val macro-F1 {m['macro_f1']:.4f}  ({epoch_seconds / 60:.1f} min)")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, round(total_loss / max(n, 1), 5), round(m["accuracy"], 5),
                                    round(m["macro_f1"], 5), opt.param_groups[0]["lr"], round(epoch_seconds, 1)])

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                        "val_metrics": m}, run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_f1": best_f1,
                    "train_seconds": train_seconds, "config": cfg}, run_dir / "last.pt")

    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    vm = best["val_metrics"]
    append_results(cfg["run_name"], "resnet18", cfg["_config_file"], "val",
                   {"accuracy": vm["accuracy"], "macro_f1": vm["macro_f1"],
                    "best_epoch": best["epoch"] + 1, "train_minutes": train_seconds / 60})
    print(f"Done. Best val macro-F1 {vm['macro_f1']:.4f} at epoch {best['epoch'] + 1}; "
          f"total training time {train_seconds / 60:.1f} min. Checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
