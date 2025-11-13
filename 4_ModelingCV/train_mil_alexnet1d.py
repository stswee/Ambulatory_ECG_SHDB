#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mil_alexnet1d_cv.py
-------------------------
Multiple-Instance Learning (MIL) pipeline for patient-level ECG classification
using 5-fold Stratified Cross-Validation with a 1D AlexNet encoder.

Each fold:
    - trains and validates on stratified subsets of patients
    - computes metrics (AUC, ACC, Precision, Recall, F1)
    - saves per-fold CSV and JSON logs
Finally:
    - reports mean ± std across folds

Data structure:
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
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, precision_recall_curve
)

# ==============================================================
# Utilities
# ==============================================================
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

# ==============================================================
# Dataset
# ==============================================================
class ECGMILDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame, segment_root: str, max_segments: Optional[int] = None):
        self.meta = meta_df.sort_values(by="patient_id").reset_index(drop=True)
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
        segments = [torch.from_numpy(np.load(f)).float().unsqueeze(0) if np.load(f).ndim == 1 else torch.from_numpy(np.load(f)).float() for f in files]
        return segments, label, pid_str

def mil_collate_fn(batch):
    segs, labels, pids = [], [], []
    for s, y, pid in batch:
        segs.append(s)
        labels.append(y)
        pids.append(pid)
    labels = torch.stack(labels)
    return segs, labels, pids

# ==============================================================
# AlexNet1D + MIL model
# ==============================================================
class AlexNet1D(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = self.features(x)
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

class MILAlexNetClassifier(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128, attn_hidden=128, clf_hidden=64, dropout=0.1):
        super().__init__()
        self.encoder = AlexNet1D(in_ch=in_ch, emb_dim=emb_dim)
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

# ==============================================================
# Config
# ==============================================================
@dataclass
class TrainConfig:
    data_root: str = "../../ECG_data"
    csv_name: str = "segmentation_with_labels.csv"
    max_segments: Optional[int] = None
    epochs: int = 10
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 4
    mixed_precision: bool = True
    n_splits: int = 5
    save_dir: str = "mil_runs_cv"
    run_name: str = "alexnet1d_mil_cv"
    seed: int = 42

# ==============================================================
# Evaluation
# ==============================================================
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for segments_bags, labels, _ in loader:
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
    return {"AUC": auc, "ACC": acc, "Precision": prec, "Recall": rec, "F1": best_f1, "Threshold": best_thr}

# ==============================================================
# Training loop (single fold)
# ==============================================================
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for segments_bags, labels, _ in loader:
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

# ==============================================================
# Main
# ==============================================================
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
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        max_segments=args.max_segments, mixed_precision=args.mixed_precision
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = os.path.join(cfg.data_root, cfg.csv_name)
    df = pd.read_csv(csv_path)
    seg_root = os.path.join(cfg.data_root, "preprocessed_segments")

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    all_fold_metrics = []

    os.makedirs(cfg.save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["outcome_label"]), 1):
        print(f"\n========== Fold {fold}/{cfg.n_splits} ==========")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        ds_train = ECGMILDataset(train_df, seg_root, cfg.max_segments)
        ds_val = ECGMILDataset(val_df, seg_root, cfg.max_segments)

        dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=mil_collate_fn)
        dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=mil_collate_fn)

        model = MILAlexNetClassifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision else None

        best_auc, best_metrics = -1, None

        for epoch in range(1, cfg.epochs + 1):
            print(f"\nEpoch {epoch}/{cfg.epochs}")
            train_loss = train_one_epoch(model, dl_train, optimizer, scaler, device)
            val_metrics = evaluate(model, dl_val, device)
            val_metrics["TrainLoss"], val_metrics["Epoch"], val_metrics["Fold"] = train_loss, epoch, fold
            print(f"  TrainLoss={train_loss:.4f} | Val={val_metrics}")

            if not math.isnan(val_metrics["AUC"]) and val_metrics["AUC"] > best_auc:
                best_auc, best_metrics = val_metrics["AUC"], val_metrics.copy()
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"{cfg.run_name}_fold{fold}_best.pt"))

        if best_metrics:
            all_fold_metrics.append(best_metrics)
            with open(os.path.join(cfg.save_dir, f"{cfg.run_name}_fold{fold}_metrics.json"), "w") as f:
                json.dump(best_metrics, f, indent=2)
            print(f"Best metrics (fold {fold}) saved.")

        # Cleanup between folds
        del model, optimizer, scaler
        torch.cuda.empty_cache()

    # Aggregate results
    if all_fold_metrics:
        df_metrics = pd.DataFrame(all_fold_metrics)
        mean_metrics = df_metrics.mean()
        std_metrics = df_metrics.std()
        summary = {k: f"{mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}" for k in ["AUC","ACC","Precision","Recall","F1"]}
        summary_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("\n===== Cross-validation Summary =====")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()
