#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mil_efficientnet1d_cv.py
------------------------------
Multiple-Instance Learning (MIL) with 1D EfficientNet-like encoder using
5-fold Stratified Cross-Validation for patient-level ECG classification.

EfficientNet1D backbone is built from:
- MBConv blocks (1D)
- Squeeze-and-Excitation (SE)
- Depthwise convolutions
"""

import os, glob, math, json, argparse
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def zpad_pid(pid: int, width: int = 3): return str(pid).zfill(width)

def list_patient_segments(root: str, pid: str):
    return sorted(glob.glob(os.path.join(root, pid, f"{pid}_window*.npy")))

# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
class ECGMILDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame, segment_root: str,
                 max_segments: Optional[int] = None):
        self.meta = meta_df.sort_values(by="patient_id").reset_index(drop=True)
        self.segment_root = segment_root
        self.max_segments = max_segments

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        pid = zpad_pid(int(row["patient_id"]))
        label = torch.tensor(float(row["outcome_label"]), dtype=torch.float32)

        files = list_patient_segments(self.segment_root, pid)
        if self.max_segments is not None and len(files) > self.max_segments:
            files = files[:self.max_segments]

        segs = []
        for f in files:
            arr = np.load(f)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            segs.append(torch.from_numpy(arr).float())

        return segs, label, pid


def mil_collate_fn(batch):
    segs, ys, pids = zip(*batch)
    return list(segs), torch.stack(list(ys)), list(pids)

# ---------------------------------------------------------------
# EfficientNet1D backbone (lightweight modification)
# ---------------------------------------------------------------
class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, reduction=8):   # CHANGED: 4 → 8
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        w = x.mean(-1, keepdim=True)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class MBConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=2, stride=1, kernel_size=7):  # CHANGED: expand_ratio 4 → 2
        super().__init__()
        mid = in_ch * expand_ratio
        padding = kernel_size // 2

        self.use_res = (in_ch == out_ch and stride == 1)

        self.expand = nn.Sequential(
            nn.Conv1d(in_ch, mid, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid),
            nn.SiLU()
        )

        self.depthwise = nn.Sequential(
            nn.Conv1d(mid, mid, kernel_size, stride, padding,
                      groups=mid, bias=False),
            nn.BatchNorm1d(mid),
            nn.SiLU()
        )

        self.se = SqueezeExcite1D(mid)

        self.project = nn.Sequential(
            nn.Conv1d(mid, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)

        if self.use_res:
            out = out + x
        return out


class EfficientNet1D(nn.Module):
    def __init__(self, in_ch=1, width_mult=1.0, emb_dim=128):
        super().__init__()

        base = int(16 * width_mult)  # CHANGED: 32 → 16

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base),
            nn.SiLU()
        )

        self.blocks = nn.Sequential(
            MBConv1D(base, base, stride=1),
            MBConv1D(base, base*2, stride=2),
            MBConv1D(base*2, base*2, stride=1),
            MBConv1D(base*2, base*4, stride=2),
            MBConv1D(base*4, base*4, stride=1),
            MBConv1D(base*4, base*4, stride=1),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base * 4, emb_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(0)

# ---------------------------------------------------------------
# Attention MIL + Classifier
# ---------------------------------------------------------------
class AttentionMIL(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, H):
        A = torch.softmax(self.w(torch.tanh(self.V(H))).squeeze(-1), dim=0)
        z = torch.sum(A.unsqueeze(-1) * H, dim=0)
        return z, A


class MILEfficientNetClassifier(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128,
                 attn_hidden=128, clf_hidden=64, dropout=0.1):
        super().__init__()
        self.encoder = EfficientNet1D(in_ch=in_ch, emb_dim=emb_dim)
        self.pool = AttentionMIL(emb_dim, attn_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, clf_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden, 1)
        )

    def forward(self, segments_list):
        feats = [self.encoder(seg) for seg in segments_list]
        H = torch.stack(feats)
        z, A = self.pool(H)
        logits = self.classifier(z).squeeze(-1)
        return logits, A

# ---------------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for bags, labels, _ in tqdm(loader, desc="Validating", leave=False):
            for segs, y in zip(bags, labels):
                segs = [s.to(device) for s in segs]
                logits, _ = model(segs)
                ps.append(torch.sigmoid(logits).item())
                ys.append(y.item())

    if len(set(ys)) < 2:
        return {m: float("nan") for m in ["AUC","ACC","Precision","Recall","F1","Threshold"]}

    auc = roc_auc_score(ys, ps)
    prec, rec, thr = precision_recall_curve(ys, ps)
    f1 = 2*prec*rec/(prec+rec+1e-8)
    i = np.argmax(f1)
    thr_opt = thr[i] if i < len(thr) else 0.5
    preds = [1 if p >= thr_opt else 0 for p in ps]

    return {
        "AUC": auc,
        "ACC": accuracy_score(ys, preds),
        "Precision": precision_score(ys, preds, zero_division=0),
        "Recall": recall_score(ys, preds, zero_division=0),
        "F1": f1[i],
        "Threshold": thr_opt
    }


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total = 0.0

    for bags, labels, _ in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        loss_sum = 0.0

        for segs, y in zip(bags, labels.to(device)):
            segs = [s.to(device) for s in segs]

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits, _ = model(segs)
                loss = crit(logits, y)

            if scaler: scaler.scale(loss).backward()
            else: loss.backward()

            loss_sum += loss.item()

        if scaler:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()

        total += loss_sum

    return total / len(loader)

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
@dataclass
class TrainConfig:
    data_root: str = "../../ECG_data"
    csv_name: str = "segmentation_with_labels.csv"
    max_segments: Optional[int] = None
    epochs: int = 10
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 0
    mixed_precision: bool = True
    save_dir: str = "mil_runs"
    run_name: str = "efficientnet1d_mil_cv"
    seed: int = 42
    n_splits: int = 5

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_segments", type=int, default=None)
    parser.add_argument("--mixed_precision", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_segments=args.max_segments,
        mixed_precision=args.mixed_precision
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = pd.read_csv(os.path.join(cfg.data_root, cfg.csv_name))
    seg_root = os.path.join(cfg.data_root, "preprocessed_segments")
    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.seed
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    fold_results = []

    for fold, (tr, val) in enumerate(skf.split(meta, meta["outcome_label"]), start=1):
        print(f"\n========== Fold {fold}/{cfg.n_splits} ==========")

        ds_tr = ECGMILDataset(meta.iloc[tr], seg_root, max_segments=cfg.max_segments)
        ds_val = ECGMILDataset(meta.iloc[val], seg_root, max_segments=cfg.max_segments)

        dl_tr = DataLoader(
            ds_tr, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=mil_collate_fn
        )
        dl_val = DataLoader(
            ds_val, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=mil_collate_fn
        )

        model = MILEfficientNetClassifier().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision else None

        best_auc, best_metrics = -1, None
        metrics_log = []

        for epoch in range(1, cfg.epochs + 1):
            print(f"\nFold {fold} | Epoch {epoch}/{cfg.epochs}")

            tr_loss = train_one_epoch(model, dl_tr, optimizer, scaler, device)
            val_m = evaluate(model, dl_val, device)
            val_m.update({"Epoch": epoch, "TrainLoss": tr_loss})
            metrics_log.append(val_m)

            print(f"  TrainLoss={tr_loss:.4f} | Val={val_m}")

            if not math.isnan(val_m["AUC"]) and val_m["AUC"] > best_auc:
                best_auc = val_m["AUC"]
                best_metrics = val_m.copy()
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        cfg.save_dir,
                        f"{cfg.run_name}_fold{fold}_best.pt"
                    )
                )
                print(f"  ↳ New best AUC: {best_auc:.4f}")

        pd.DataFrame(metrics_log).to_csv(
            os.path.join(cfg.save_dir,
            f"{cfg.run_name}_fold{fold}_metrics.csv"),
            index=False
        )

        if best_metrics:
            with open(os.path.join(
                cfg.save_dir,
                f"{cfg.run_name}_fold{fold}_best_metrics.json"
            ), "w") as f:
                json.dump(best_metrics, f, indent=2)

            fold_results.append(best_metrics)

        del model
        torch.cuda.empty_cache()

    summary = pd.DataFrame(fold_results)
    mean = summary.mean(numeric_only=True)
    std = summary.std(numeric_only=True)
    summary.loc["Mean"] = mean
    summary.loc["Std"] = std

    csv_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_cv_summary.csv")
    summary.to_csv(csv_path)

    print("\n===== Cross-Validation Summary =====")
    for m in ["AUC","ACC","Precision","Recall","F1"]:
        print(f"{m:>10}: {mean[m]:.4f} ± {std[m]:.4f}")
    print("Summary CSV saved to:", csv_path)


if __name__ == "__main__":
    main()
