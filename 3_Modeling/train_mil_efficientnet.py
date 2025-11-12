#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mil_efficientnet.py
-------------------------
Multiple-Instance Learning (MIL) pipeline for patient-level ECG classification
(Healthy vs Pathological) using 30s segments as instances.

This version uses a 1D EfficientNet-like encoder (MBConv blocks)
instead of ResNet1D.

All other components (MIL attention, loss, metrics, logging) remain identical.

ECG_data lives *outside* Ambulatory_ECG_SHDB:
    ../../ECG_data/segmentation_with_labels.csv
    ../../ECG_data/preprocessed_segments/
"""

import os
import glob
import math
import argparse
import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve


# ---------------------------------------------------------------
# Utility
# ---------------------------------------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zpad_pid(pid: int, width: int = 3):
    return str(pid).zfill(width)


def list_patient_segments(segment_root: str, pid_str: str):
    return sorted(glob.glob(os.path.join(segment_root, pid_str, f"{pid_str}_window*.npy")))


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
class ECGMILDataset(Dataset):
    def __init__(self, csv_path: str, segment_root: str, split: Optional[str] = None, max_segments: Optional[int] = None):
        self.meta = pd.read_csv(csv_path)
        if split is not None:
            self.meta = self.meta[self.meta["split"].str.lower() == split.lower()]
        self.meta = self.meta.sort_values(by="patient_id").reset_index(drop=True)
        self.segment_root = segment_root
        self.max_segments = max_segments

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        pid_str = zpad_pid(int(row["patient_id"]))
        label = torch.tensor(float(row["outcome_label"]), dtype=torch.float32)
        files = list_patient_segments(self.segment_root, pid_str)
        if self.max_segments and len(files) > self.max_segments:
            idxs = np.linspace(0, len(files) - 1, self.max_segments, dtype=int)
            files = [files[i] for i in idxs]
        segments = []
        for f in files:
            arr = np.load(f)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            segments.append(torch.from_numpy(arr).float())
        return segments, label, pid_str


def mil_collate_fn(batch):
    segs, labels, pids = [], [], []
    for s, y, pid in batch:
        segs.append(s)
        labels.append(y)
        pids.append(pid)
    labels = torch.stack(labels)
    return segs, labels, pids


# ---------------------------------------------------------------
# EfficientNet-1D encoder
# ---------------------------------------------------------------
class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv1d(channels, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, channels, 1)

    def forward(self, x):
        s = x.mean(-1, keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class MBConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=4, stride=1):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = (in_ch == out_ch) and (stride == 1)
        self.expand = nn.Conv1d(in_ch, hidden, 1, bias=False)
        self.bn0 = nn.BatchNorm1d(hidden)
        self.dwconv = nn.Conv1d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.se = SqueezeExcite1D(hidden)
        self.project = nn.Conv1d(hidden, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        out = F.silu(self.bn0(self.expand(x)))
        out = F.silu(self.bn1(self.dwconv(out)))
        out = self.se(out)
        out = self.bn2(self.project(out))
        if self.use_res:
            out += x
        return out


class EfficientNet1D(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128, width_mult=1.0):
        super().__init__()
        base_channels = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.SiLU()
        )
        cfg = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 40, 2, 2),
            (6, 80, 3, 2),
            (6, 112, 3, 1),
            (6, 192, 4, 2),
            (6, 320, 1, 1),
        ]
        layers = []
        in_c = base_channels
        for t, c, n, s in cfg:
            out_c = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(MBConv1D(in_c, out_c, expand_ratio=t, stride=stride))
                in_c = out_c
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv1d(in_c, 1280, 1, bias=False),
            nn.BatchNorm1d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(1280, emb_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x).squeeze(-1)
        x = self.fc(x)
        return x.squeeze(0)


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
    def __init__(self, in_ch=1, emb_dim=128, attn_hidden=128, clf_hidden=64, dropout=0.1):
        super().__init__()
        self.encoder = EfficientNet1D(in_ch=in_ch, emb_dim=emb_dim)
        self.pool = AttentionMIL(emb_dim, attn_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, clf_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden, 1),
        )

    def forward(self, segments_list):
        feats = [self.encoder(seg) for seg in segments_list]
        H = torch.stack(feats)
        z, A = self.pool(H)
        logits = self.classifier(z).squeeze(-1)
        return logits, A


# ---------------------------------------------------------------
# Training + Eval utilities
# ---------------------------------------------------------------
@dataclass
class TrainConfig:
    data_root: str = "../../ECG_data"
    csv_name: str = "segmentation_with_labels.csv"
    split_train: str = "train"
    split_val: str = "val"
    max_segments: Optional[int] = None
    epochs: int = 10
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 4
    mixed_precision: bool = True
    save_dir: str = "mil_runs"
    run_name: str = "efficientnet1d_mil"
    seed: int = 42


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for segments_bags, labels, _ in tqdm(loader, desc="Validating", leave=False):
            for segments, y in zip(segments_bags, labels):
                segments = [s.to(device) for s in segments]
                logits, _ = model(segments)
                p = torch.sigmoid(logits)
                ys.append(y.item())
                ps.append(float(p.item()))
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    precision, recall, thresholds = precision_recall_curve(ys, ps)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    preds = [1 if p >= best_thr else 0 for p in ps]
    acc = accuracy_score(ys, preds)
    prec = precision_score(ys, preds, zero_division=0)
    rec = recall_score(ys, preds, zero_division=0)
    return {
        "AUC": auc,
        "ACC": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": best_f1,
        "Threshold": best_thr
    }


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for segments_bags, labels, _ in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        loss_sum = 0.0
        for segments, y in zip(segments_bags, labels.to(device)):
            segments = [s.to(device) for s in segments]
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits, _ = model(segments)
                loss = criterion(logits, y)
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_sum += loss.item()
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        total_loss += loss_sum
    return total_loss / len(loader)


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
        mixed_precision=args.mixed_precision,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = os.path.join(cfg.data_root, cfg.csv_name)
    seg_root = os.path.join(cfg.data_root, "preprocessed_segments")
    ds_train = ECGMILDataset(csv_path, seg_root, split=cfg.split_train, max_segments=cfg.max_segments)
    ds_val = ECGMILDataset(csv_path, seg_root, split=cfg.split_val, max_segments=cfg.max_segments)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=mil_collate_fn)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=mil_collate_fn)

    model = MILEfficientNetClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision else None

    os.makedirs(cfg.save_dir, exist_ok=True)
    metrics_log, best_metrics, best_auc = [], None, -1

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_loss = train_one_epoch(model, dl_train, optimizer, scaler, device)
        val_metrics = evaluate(model, dl_val, device)
        val_metrics["TrainLoss"], val_metrics["Epoch"] = train_loss, epoch
        metrics_log.append(val_metrics)
        print(f"  TrainLoss={train_loss:.4f} | Val={val_metrics}")
        if not math.isnan(val_metrics["AUC"]) and val_metrics["AUC"] > best_auc:
            best_auc, best_metrics = val_metrics["AUC"], val_metrics.copy()
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"{cfg.run_name}_best.pt"))
            print(f"  ↳ New best AUC: {best_auc:.4f}")

    pd.DataFrame(metrics_log).to_csv(os.path.join(cfg.save_dir, f"{cfg.run_name}_metrics.csv"), index=False)
    if best_metrics:
        with open(os.path.join(cfg.save_dir, f"{cfg.run_name}_best_metrics.json"), "w") as f:
            json.dump(best_metrics, f, indent=2)
    print("Training complete.")


if __name__ == "__main__":
    main()
