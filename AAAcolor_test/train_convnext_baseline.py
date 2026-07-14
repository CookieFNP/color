from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


class ColorCropDataset(Dataset):
    def __init__(self, labels_csv: Path, dataset_dir: Path, split: str, transform=None, source: str = "any"):
        self.dataset_dir = dataset_dir
        self.transform = transform

        df = pd.read_csv(labels_csv, encoding="utf-8-sig")
        if "split" not in df.columns:
            raise RuntimeError("labels.csv 里没有 split 列，请先用 build_convnext_dataset.py 生成数据集。")

        df = df[df["split"] == split].copy()
        if source != "any" and "source" in df.columns:
            df = df[df["source"] == source].copy()

        if len(df) == 0:
            raise RuntimeError(f"split={split}, source={source} 没有样本。")

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.dataset_dir / str(row["image_path"])
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        y = int(row["label_id"])
        meta = {
            "run": str(row.get("run", "")),
            "code": str(row.get("code", "")),
            "name": str(row.get("name", "")),
            "image_path": str(row.get("image_path", "")),
        }
        return img, y, meta


def build_transforms(image_size: int):
    # 颜色识别任务不能做 hue shift。这里增强偏保守：模拟亮度、对比度、轻微饱和度、裁剪、旋转。
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.86, 1.0), ratio=(0.92, 1.08)),
        transforms.RandomRotation(degrees=5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.14, contrast=0.10, saturation=0.05, hue=0.0)
        ], p=0.75),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
        ], p=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.08, scale=(0.01, 0.04), ratio=(0.3, 3.3), value="random"),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    backbone = backbone.lower().strip()

    if backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
    elif backbone == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = models.convnext_small(weights=weights)
    elif backbone == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        model = models.convnext_base(weights=weights)
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"不支持 backbone: {backbone}")

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool, backbone: str) -> None:
    # 只冻结特征提取层，保留分类头训练。
    for p in model.parameters():
        p.requires_grad = trainable

    if not trainable:
        if backbone.startswith("convnext"):
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif backbone.startswith("resnet"):
            for p in model.fc.parameters():
                p.requires_grad = True


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1, 3, 5)) -> dict[str, float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    out = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        out[f"top{k}"] = float(correct_k.item() / target.size(0))
    return out


def run_one_epoch(model, loader, criterion, optimizer, device, scaler, train: bool, amp: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_n = 0
    correct1 = 0
    correct3 = 0
    correct5 = 0

    all_preds = []

    for images, labels, metas in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if device.type == "cuda" and amp:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda" and amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        acc = topk_accuracy(logits.detach(), labels.detach(), topk=(1, 3, 5))
        correct1 += acc["top1"] * bs
        correct3 += acc["top3"] * bs
        correct5 += acc["top5"] * bs

        if not train:
            probs = torch.softmax(logits.detach(), dim=1)
            top_prob, top_idx = probs.topk(5, dim=1)
            for i in range(bs):
                row = {
                    "run": metas["run"][i],
                    "code": metas["code"][i],
                    "name": metas["name"][i],
                    "image_path": metas["image_path"][i],
                    "true_label": int(labels[i].detach().cpu().item()),
                }
                for k in range(5):
                    row[f"top{k+1}_label"] = int(top_idx[i, k].detach().cpu().item())
                    row[f"top{k+1}_prob"] = float(top_prob[i, k].detach().cpu().item())
                all_preds.append(row)

    metrics = {
        "loss": total_loss / max(total_n, 1),
        "top1": correct1 / max(total_n, 1),
        "top3": correct3 / max(total_n, 1),
        "top5": correct5 / max(total_n, 1),
        "n": total_n,
    }
    return metrics, all_preds


def load_code_map(labels_csv: Path) -> dict[int, dict[str, str]]:
    df = pd.read_csv(labels_csv, encoding="utf-8-sig")
    out = {}
    for _, r in df.iterrows():
        lid = int(r["label_id"])
        if lid not in out:
            out[lid] = {"code": str(r["code"]), "name": str(r.get("name", ""))}
    return out


def enrich_predictions(pred_rows: list[dict[str, Any]], code_map: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    for r in pred_rows:
        for k in range(1, 6):
            lid = int(r[f"top{k}_label"])
            r[f"top{k}_code"] = code_map.get(lid, {}).get("code", f"W{lid+1:03d}")
            r[f"top{k}_name"] = code_map.get(lid, {}).get("name", "")
        true_lid = int(r["true_label"])
        r["true_code_from_label"] = code_map.get(true_lid, {}).get("code", f"W{true_lid+1:03d}")
        r["top1_correct"] = int(r["top1_label"] == r["true_label"])
        r["top3_correct"] = int(any(r[f"top{k}_label"] == r["true_label"] for k in range(1, 4)))
        r["top5_correct"] = int(any(r[f"top{k}_label"] == r["true_label"] for k in range(1, 6)))
    return pred_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 ConvNeXt 纯图像 128 类颜色分类 baseline。")
    ap.add_argument("--dataset", default="color_cls_dataset")
    ap.add_argument("--labels", default="", help="默认使用 <dataset>/labels.csv")
    ap.add_argument("--out", default="convnext_tiny_baseline_out")
    ap.add_argument("--backbone", default="convnext_tiny",
                    choices=["convnext_tiny", "convnext_small", "convnext_base", "resnet50"])
    ap.add_argument("--num-classes", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=0, help="Windows 下先用 0，稳定后可改 2/4。")
    ap.add_argument("--source", default="any", choices=["any", "original", "root"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--freeze-epochs", type=int, default=2, help="前几轮只训练分类头，数据少时更稳。")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", action="store_true", help="CUDA 混合精度，显卡支持时可开。")
    args = ap.parse_args()

    seed_everything(args.seed)

    dataset_dir = Path(args.dataset)
    labels_csv = Path(args.labels) if args.labels else dataset_dir / "labels.csv"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    train_tf, val_tf = build_transforms(args.image_size)

    train_ds = ColorCropDataset(labels_csv, dataset_dir, "train", transform=train_tf, source=args.source)
    val_ds = ColorCropDataset(labels_csv, dataset_dir, "val", transform=val_tf, source=args.source)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    print("train samples:", len(train_ds))
    print("val samples:", len(val_ds))

    model = build_model(args.backbone, args.num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    code_map = load_code_map(labels_csv)

    history = []
    best_top1 = -1.0
    best_epoch = -1

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    })
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if epoch <= args.freeze_epochs:
            set_backbone_trainable(model, False, args.backbone)
        elif epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, True, args.backbone)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.epochs - args.freeze_epochs, 1)
            )

        train_m, _ = run_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, train=True, amp=args.amp
        )
        val_m, val_preds = run_one_epoch(
            model, val_loader, criterion, optimizer, device, scaler, train=False, amp=args.amp
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_m["loss"],
            "train_top1": train_m["top1"],
            "train_top3": train_m["top3"],
            "train_top5": train_m["top5"],
            "val_loss": val_m["loss"],
            "val_top1": val_m["top1"],
            "val_top3": val_m["top3"],
            "val_top5": val_m["top5"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        write_csv(out_dir / "metrics.csv", history)

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train top1={train_m['top1']:.4f} top3={train_m['top3']:.4f} loss={train_m['loss']:.4f} | "
            f"val top1={val_m['top1']:.4f} top3={val_m['top3']:.4f} top5={val_m['top5']:.4f} loss={val_m['loss']:.4f} | "
            f"{row['seconds']:.1f}s"
        )

        # 保存 last
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "backbone": args.backbone,
            "num_classes": args.num_classes,
            "code_map": code_map,
            "config": config,
            "val_metrics": val_m,
        }, out_dir / "last.pt")

        if val_m["top1"] > best_top1:
            best_top1 = val_m["top1"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "backbone": args.backbone,
                "num_classes": args.num_classes,
                "code_map": code_map,
                "config": config,
                "val_metrics": val_m,
            }, out_dir / "best.pt")

            val_preds = enrich_predictions(val_preds, code_map)
            write_csv(out_dir / "val_predictions_best.csv", val_preds)

    summary = {
        "best_epoch": best_epoch,
        "best_val_top1": best_top1,
        "last_epoch": args.epochs,
        "out": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print("best epoch:", best_epoch)
    print("best val top1:", best_top1)
    print("out:", out_dir.resolve())
    print("metrics:", out_dir / "metrics.csv")
    print("best model:", out_dir / "best.pt")
    print("best predictions:", out_dir / "val_predictions_best.csv")


if __name__ == "__main__":
    main()
