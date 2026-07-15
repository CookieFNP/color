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
from torchvision import transforms, models
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp.*")

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
    fields = []
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


def choose_numeric_features(df: pd.DataFrame) -> list[str]:
    """
    只选择推理时也能拿到的数值特征。
    不把 code / label_id / std_Lab / deltaE / pred_code 等答案信息喂给模型。
    """
    preferred = [
        "raw_L", "raw_a", "raw_b",
        "root_L", "root_a", "root_b",
        "final_L", "final_a", "final_b",
        "local_bg_L", "local_bg_a", "local_bg_b",
        "crop_R_mean", "crop_G_mean", "crop_B_mean",
        "crop_R_std", "crop_G_std", "crop_B_std",
        "crop_H_mean", "crop_S_mean", "crop_V_mean",
        "crop_H_std", "crop_S_std", "crop_V_std",
    ]
    out = []
    for c in preferred:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
            out.append(c)
    return out


def fit_scaler(df: pd.DataFrame, cols: list[str]):
    arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return mean, std


def delta_e_ciede2000(lab1, lab2) -> float:
    L1, a1, b1 = [float(x) for x in lab1]
    L2, a2, b2 = [float(x) for x in lab2]
    C1 = math.sqrt(a1*a1 + b1*b1)
    C2 = math.sqrt(a2*a2 + b2*b2)
    avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - math.sqrt(avg_C**7 / (avg_C**7 + 25**7))) if avg_C != 0 else 0.0
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.sqrt(a1p*a1p + b1*b1)
    C2p = math.sqrt(a2p*a2p + b2*b2)

    def hp(ap, b):
        if ap == 0 and b == 0:
            return 0.0
        h = math.degrees(math.atan2(b, ap))
        return h + 360 if h < 0 else h

    h1p = hp(a1p, b1)
    h2p = hp(a2p, b2)
    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        dhp = dh

    dHp = 2 * math.sqrt(C1p*C2p) * math.sin(math.radians(dhp / 2.0))
    avg_Lp = (L1 + L2) / 2.0
    avg_Cp = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        avg_hp = h1p + h2p
    else:
        dh_abs = abs(h1p - h2p)
        if dh_abs <= 180:
            avg_hp = (h1p + h2p) / 2.0
        elif h1p + h2p < 360:
            avg_hp = (h1p + h2p + 360) / 2.0
        else:
            avg_hp = (h1p + h2p - 360) / 2.0

    T = (
        1
        - 0.17 * math.cos(math.radians(avg_hp - 30))
        + 0.24 * math.cos(math.radians(2 * avg_hp))
        + 0.32 * math.cos(math.radians(3 * avg_hp + 6))
        - 0.20 * math.cos(math.radians(4 * avg_hp - 63))
    )
    delta_theta = 30 * math.exp(-(((avg_hp - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(avg_Cp**7 / (avg_Cp**7 + 25**7)) if avg_Cp != 0 else 0.0
    Sl = 1 + (0.015 * ((avg_Lp - 50) ** 2)) / math.sqrt(20 + ((avg_Lp - 50) ** 2))
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -math.sin(math.radians(2 * delta_theta)) * Rc
    return float(math.sqrt((dLp/Sl)**2 + (dCp/Sc)**2 + (dHp/Sh)**2 + Rt*(dCp/Sc)*(dHp/Sh)))


def de_stats(pred_lab: np.ndarray, std_lab: np.ndarray, base_lab: np.ndarray):
    pred_de = np.array([delta_e_ciede2000(p, s) for p, s in zip(pred_lab, std_lab)])
    base_de = np.array([delta_e_ciede2000(b, s) for b, s in zip(base_lab, std_lab)])
    return {
        "pred_de_mean": float(pred_de.mean()),
        "pred_de_median": float(np.median(pred_de)),
        "pred_de_p95": float(np.percentile(pred_de, 95)),
        "pred_de_max": float(pred_de.max()),
        "base_de_mean": float(base_de.mean()),
        "base_de_median": float(np.median(base_de)),
        "base_de_p95": float(np.percentile(base_de, 95)),
        "base_de_max": float(base_de.max()),
    }


class ColorHybridDataset(Dataset):
    def __init__(self, labels_csv, dataset_dir, split, feature_cols, feat_mean, feat_std, transform, base_prefix="final"):
        self.dataset_dir = Path(dataset_dir)
        df = pd.read_csv(labels_csv, encoding="utf-8-sig")
        df = df[df["split"] == split].copy()
        if len(df) == 0:
            raise RuntimeError(f"split={split} 没有样本")
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.feat_mean = feat_mean.astype(np.float32)
        self.feat_std = feat_std.astype(np.float32)
        self.transform = transform
        self.base_prefix = base_prefix

        needed = ["image_path", "label_id", "std_L", "std_a", "std_b",
                  f"{base_prefix}_L", f"{base_prefix}_a", f"{base_prefix}_b"]
        for c in needed:
            if c not in self.df.columns:
                raise RuntimeError(f"labels.csv 缺少列: {c}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(self.dataset_dir / str(r["image_path"])).convert("RGB")
        if self.transform:
            img = self.transform(img)

        feat = pd.to_numeric(r[self.feature_cols], errors="coerce").to_numpy(dtype=np.float32)
        feat = np.where(np.isfinite(feat), feat, self.feat_mean)
        feat = (feat - self.feat_mean) / self.feat_std

        base_lab = np.array([r[f"{self.base_prefix}_L"], r[f"{self.base_prefix}_a"], r[f"{self.base_prefix}_b"]], dtype=np.float32)
        std_lab = np.array([r["std_L"], r["std_a"], r["std_b"]], dtype=np.float32)

        meta = {
            "run": str(r.get("run", "")),
            "code": str(r.get("code", "")),
            "name": str(r.get("name", "")),
            "image_path": str(r.get("image_path", "")),
        }
        return img, torch.tensor(feat), torch.tensor(int(r["label_id"])), torch.tensor(base_lab), torch.tensor(std_lab), meta


def build_transforms(size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.86, 1.0), ratio=(0.92, 1.08)),
        transforms.RandomRotation(5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.14, contrast=0.10, saturation=0.05, hue=0.0)], p=0.75),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.08, scale=(0.01, 0.04), ratio=(0.3, 3.3), value="random"),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_backbone(name, pretrained=True):
    name = name.lower()
    if name == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = models.convnext_tiny(weights=w)
    elif name == "convnext_small":
        w = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        m = models.convnext_small(weights=w)
    elif name == "convnext_base":
        w = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        m = models.convnext_base(weights=w)
    else:
        raise ValueError(name)
    dim = m.classifier[-1].in_features
    m.classifier[-1] = nn.Identity()
    return m, dim


class HybridModel(nn.Module):
    def __init__(self, backbone_name, num_numeric, num_classes=128, pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone, img_dim = build_backbone(backbone_name, pretrained)
        self.num_mlp = nn.Sequential(
            nn.Linear(num_numeric, 128), nn.GELU(), nn.BatchNorm1d(128), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + 128, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(512, num_classes)
        self.lab_head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 3))
        nn.init.zeros_(self.lab_head[-1].weight)
        nn.init.zeros_(self.lab_head[-1].bias)

    def forward(self, image, numeric):
        img_feat = self.backbone(image)
        num_feat = self.num_mlp(numeric)
        feat = self.fusion(torch.cat([img_feat, num_feat], dim=1))
        logits = self.cls_head(feat)
        delta = self.lab_head(feat)
        return logits, delta


def set_backbone_trainable(model, flag):
    for p in model.backbone.parameters():
        p.requires_grad = flag


def topk_acc(logits, y):
    out = {}
    _, pred = logits.topk(5, dim=1)
    for k in (1, 3, 5):
        out[f"top{k}"] = float((pred[:, :k] == y[:, None]).any(dim=1).float().mean().item())
    return out


def run_epoch(model, loader, ce_loss, lab_loss, optimizer, device, scaler, train, amp, lambda_lab):
    model.train(train)
    total_n = 0
    loss_sum = cls_sum = lab_sum = 0.0
    acc_sum = {"top1": 0.0, "top3": 0.0, "top5": 0.0}
    pred_labs, std_labs, base_labs = [], [], []
    rows = []

    for img, feat, y, base_lab, std_lab, meta in loader:
        img = img.to(device, non_blocking=True)
        feat = feat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        base_lab = base_lab.to(device, non_blocking=True)
        std_lab = std_lab.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if device.type == "cuda" and amp:
                with torch.cuda.amp.autocast():
                    logits, delta = model(img, feat)
                    pred_lab = base_lab + delta
                    cls_l = ce_loss(logits, y)
                    lab_l = lab_loss(pred_lab, std_lab)
                    loss = cls_l + lambda_lab * lab_l
            else:
                logits, delta = model(img, feat)
                pred_lab = base_lab + delta
                cls_l = ce_loss(logits, y)
                lab_l = lab_loss(pred_lab, std_lab)
                loss = cls_l + lambda_lab * lab_l

            if train:
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda" and amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        bs = y.size(0)
        total_n += bs
        loss_sum += float(loss.item()) * bs
        cls_sum += float(cls_l.item()) * bs
        lab_sum += float(lab_l.item()) * bs
        a = topk_acc(logits.detach(), y.detach())
        for k in acc_sum:
            acc_sum[k] += a[k] * bs

        pred_np = pred_lab.detach().float().cpu().numpy()
        std_np = std_lab.detach().float().cpu().numpy()
        base_np = base_lab.detach().float().cpu().numpy()
        pred_labs.append(pred_np)
        std_labs.append(std_np)
        base_labs.append(base_np)

        if not train:
            prob = torch.softmax(logits.detach(), dim=1)
            top_prob, top_idx = prob.topk(5, dim=1)
            delta_np = delta.detach().float().cpu().numpy()
            for i in range(bs):
                row = {
                    "run": meta["run"][i],
                    "code": meta["code"][i],
                    "name": meta["name"][i],
                    "image_path": meta["image_path"][i],
                    "true_label": int(y[i].detach().cpu().item()),
                    "base_L": float(base_np[i, 0]), "base_a": float(base_np[i, 1]), "base_b": float(base_np[i, 2]),
                    "std_L": float(std_np[i, 0]), "std_a": float(std_np[i, 1]), "std_b": float(std_np[i, 2]),
                    "pred_delta_L": float(delta_np[i, 0]), "pred_delta_a": float(delta_np[i, 1]), "pred_delta_b": float(delta_np[i, 2]),
                    "pred_L": float(pred_np[i, 0]), "pred_a": float(pred_np[i, 1]), "pred_b": float(pred_np[i, 2]),
                    "base_deltaE": delta_e_ciede2000(base_np[i], std_np[i]),
                    "pred_deltaE": delta_e_ciede2000(pred_np[i], std_np[i]),
                }
                for k in range(5):
                    row[f"top{k+1}_label"] = int(top_idx[i, k].detach().cpu().item())
                    row[f"top{k+1}_prob"] = float(top_prob[i, k].detach().cpu().item())
                row["top1_correct"] = int(row["top1_label"] == row["true_label"])
                row["top3_correct"] = int(any(row[f"top{k}_label"] == row["true_label"] for k in range(1, 4)))
                row["top5_correct"] = int(any(row[f"top{k}_label"] == row["true_label"] for k in range(1, 6)))
                rows.append(row)

    pred_all = np.concatenate(pred_labs, axis=0)
    std_all = np.concatenate(std_labs, axis=0)
    base_all = np.concatenate(base_labs, axis=0)
    d = de_stats(pred_all, std_all, base_all)
    m = {
        "loss": loss_sum / max(total_n, 1),
        "cls_loss": cls_sum / max(total_n, 1),
        "lab_loss": lab_sum / max(total_n, 1),
        "top1": acc_sum["top1"] / max(total_n, 1),
        "top3": acc_sum["top3"] / max(total_n, 1),
        "top5": acc_sum["top5"] / max(total_n, 1),
        "n": total_n,
        **d,
    }
    return m, rows


def load_code_map(labels_csv):
    df = pd.read_csv(labels_csv, encoding="utf-8-sig")
    m = {}
    for _, r in df.iterrows():
        lid = int(r["label_id"])
        if lid not in m:
            m[lid] = {"code": str(r["code"]), "name": str(r.get("name", ""))}
    return m


def enrich(rows, code_map):
    for r in rows:
        for k in range(1, 6):
            lid = int(r[f"top{k}_label"])
            r[f"top{k}_code"] = code_map.get(lid, {}).get("code", f"W{lid+1:03d}")
            r[f"top{k}_name"] = code_map.get(lid, {}).get("name", "")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Hybrid ConvNeXt：128类分类 + Lab residual校正。")
    ap.add_argument("--dataset", default="color_cls_dataset")
    ap.add_argument("--labels", default="")
    ap.add_argument("--out", default="hybrid_convnext_lab_out")
    ap.add_argument("--backbone", default="convnext_tiny", choices=["convnext_tiny", "convnext_small", "convnext_base"])
    ap.add_argument("--num-classes", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--lambda-lab", type=float, default=0.2)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--base-prefix", default="final", choices=["final", "root", "raw"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    dataset_dir = Path(args.dataset)
    labels_csv = Path(args.labels) if args.labels else dataset_dir / "labels.csv"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv, encoding="utf-8-sig")
    train_df = df[df["split"] == "train"].copy()
    feature_cols = choose_numeric_features(df)
    feat_mean, feat_std = fit_scaler(train_df, feature_cols)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
    print("features:", feature_cols)

    train_tf, val_tf = build_transforms(args.image_size)
    train_ds = ColorHybridDataset(labels_csv, dataset_dir, "train", feature_cols, feat_mean, feat_std, train_tf, args.base_prefix)
    val_ds = ColorHybridDataset(labels_csv, dataset_dir, "val", feature_cols, feat_mean, feat_std, val_tf, args.base_prefix)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type=="cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type=="cuda")

    print("train:", len(train_ds), "val:", len(val_ds))

    model = HybridModel(args.backbone, len(feature_cols), args.num_classes, pretrained=not args.no_pretrained).to(device)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    lab_loss = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda" and args.amp))
    code_map = load_code_map(labels_csv)

    config = vars(args).copy()
    config.update({
        "feature_cols": feature_cols,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "device": str(device),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    })
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    history = []
    best_de = 1e18
    best_acc = -1
    best_combo = -1e18

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if epoch <= args.freeze_epochs:
            set_backbone_trainable(model, False)
        elif epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.freeze_epochs, 1))

        train_m, _ = run_epoch(model, train_loader, ce_loss, lab_loss, optimizer, device, scaler, True, args.amp, args.lambda_lab)
        val_m, val_rows = run_epoch(model, val_loader, ce_loss, lab_loss, optimizer, device, scaler, False, args.amp, args.lambda_lab)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_m["loss"],
            "train_top1": train_m["top1"],
            "train_top3": train_m["top3"],
            "train_pred_de_mean": train_m["pred_de_mean"],
            "train_base_de_mean": train_m["base_de_mean"],
            "val_loss": val_m["loss"],
            "val_cls_loss": val_m["cls_loss"],
            "val_lab_loss": val_m["lab_loss"],
            "val_top1": val_m["top1"],
            "val_top3": val_m["top3"],
            "val_top5": val_m["top5"],
            "val_pred_de_mean": val_m["pred_de_mean"],
            "val_pred_de_median": val_m["pred_de_median"],
            "val_pred_de_p95": val_m["pred_de_p95"],
            "val_pred_de_max": val_m["pred_de_max"],
            "val_base_de_mean": val_m["base_de_mean"],
            "val_base_de_median": val_m["base_de_median"],
            "val_base_de_p95": val_m["base_de_p95"],
            "val_base_de_max": val_m["base_de_max"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        write_csv(out_dir / "metrics.csv", history)

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train top1={train_m['top1']:.4f} de={train_m['pred_de_mean']:.3f}/{train_m['base_de_mean']:.3f} | "
            f"val top1={val_m['top1']:.4f} top3={val_m['top3']:.4f} "
            f"de={val_m['pred_de_mean']:.3f}/{val_m['base_de_mean']:.3f} "
            f"p95={val_m['pred_de_p95']:.3f} max={val_m['pred_de_max']:.3f} | "
            f"{row['seconds']:.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "backbone": args.backbone,
            "num_classes": args.num_classes,
            "feature_cols": feature_cols,
            "feature_mean": feat_mean.tolist(),
            "feature_std": feat_std.tolist(),
            "base_prefix": args.base_prefix,
            "code_map": code_map,
            "config": config,
            "val_metrics": val_m,
        }
        torch.save(ckpt, out_dir / "last.pt")
        val_rows = enrich(val_rows, code_map)

        if val_m["pred_de_mean"] < best_de:
            best_de = val_m["pred_de_mean"]
            torch.save(ckpt, out_dir / "best_de.pt")
            write_csv(out_dir / "val_predictions_best_de.csv", val_rows)

        if val_m["top1"] > best_acc:
            best_acc = val_m["top1"]
            torch.save(ckpt, out_dir / "best_acc.pt")
            write_csv(out_dir / "val_predictions_best_acc.csv", val_rows)

        combo = val_m["top1"] - 0.02 * val_m["pred_de_mean"]
        if combo > best_combo:
            best_combo = combo
            torch.save(ckpt, out_dir / "best_combo.pt")
            write_csv(out_dir / "val_predictions_best_combo.csv", val_rows)

    summary = {
        "best_val_top1": best_acc,
        "best_val_pred_de_mean": best_de,
        "best_combo_score": best_combo,
        "out": str(out_dir),
        "note": "best_de.pt追求低ΔE；best_acc.pt追求分类准确率；best_combo.pt折中。",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Done ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
