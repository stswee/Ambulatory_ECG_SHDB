#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mil_resnet1d.py
---------------------
Multiple-Instance Learning (MIL) pipeline for patient-level ECG classification
(Healthy vs Pathological) using 30s segments as instances.

Now includes:
- Saving per-epoch metrics (AUC, ACC, Precision, Recall, F1, Threshold)
- Saving best model metrics
- Optimal threshold computation for F1
- JSON + CSV logging

ECG_data lives *outside* Ambulatory_ECG_SHDB:
    ../../ECG_data/segmentation_with_labels.csv
    ../../ECG_data/preprocessed_segments/
"""

import os
import glob
import math
import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ---------------------------------------------------------------
# Utility Functions
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
# Dataset Definition
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
# Model
# ---------------------------------------------------------------
class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        return F.relu(out + identity)


class ResNet1D(nn.Module):
    def __init__(self, in_ch=1, base=64, emb_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, 2, 3, bias=False),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )
        self.layer1 = self._make_layer(base, base, 2, 1)
        self.layer2 = self._make_layer(base, base * 2, 2, 2)
        self.layer3 = self._make_layer(base * 2, base * 4, 2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base * 4, emb_dim)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(0)


class AttentionMIL(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, H):
        A = torch.softmax(self.w(torch.tanh(self.V(H))).squeeze(-1), dim=0)
        z = torch.sum(A.unsqueeze(-1) * H, dim=0)
        return z, A


class MILResNetClassifier(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128, attn_hidden=128, clf_hidden=64, dropout=0.1):
        super().__init__()
        self.encoder = ResNet1D(in_ch=in_ch, emb_dim=emb_dim)
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
# Config
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
    run_name: str = "resnet1d_mil"
    seed: int = 42


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
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

    if SKLEARN_OK and len(set(ys)) > 1:
        auc = roc_auc_score(ys, ps)
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
    else:
        preds = [1 if p >= 0.5 else 0 for p in ps]
        acc = sum(p == y for p, y in zip(preds, ys)) / len(ys)
        return {"AUC": float("nan"), "ACC": acc, "Precision": 0, "Recall": 0, "F1": 0, "Threshold": 0.5}


# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------
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

    model = MILResNetClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision else None

    os.makedirs(cfg.save_dir, exist_ok=True)
    metrics_log = []
    best_metrics = None
    best_auc = -1

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_loss = train_one_epoch(model, dl_train, optimizer, scaler, device)
        val_metrics = evaluate(model, dl_val, device)
        val_metrics["TrainLoss"] = train_loss
        val_metrics["Epoch"] = epoch
        metrics_log.append(val_metrics)

        print(f"  TrainLoss={train_loss:.4f} | Val={val_metrics}")
        if SKLEARN_OK and not math.isnan(val_metrics["AUC"]) and val_metrics["AUC"] > best_auc:
            best_auc = val_metrics["AUC"]
            best_metrics = val_metrics.copy()
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"{cfg.run_name}_best.pt"))
            print(f"  ↳ New best AUC: {best_auc:.4f}")

    # Save metrics
    metrics_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_metrics.csv")
    pd.DataFrame(metrics_log).to_csv(metrics_path, index=False)
    if best_metrics:
        best_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_best_metrics.json")
        with open(best_path, "w") as f:
            json.dump(best_metrics, f, indent=2)
        print(f"\nBest model metrics saved to: {best_path}")
    print(f"Full training log saved to: {metrics_path}")


if __name__ == "__main__":
    main()
