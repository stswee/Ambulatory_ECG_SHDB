#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mil_alexnet1d_cv_fft.py
----------------------------------
Multiple-Instance Learning (MIL) with 1D AlexNet encoder using
5-fold Stratified Cross-Validation for patient-level ECG classification.

This version uses FOURIER amplitude data:
    ../../ECG_data/fft_amplitude_segments/{pid}/{pid}_windowXXXX_amp.npy
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def zpad_pid(pid: int, width: int = 3):
    return str(pid).zfill(width)

def list_patient_segments(root: str, pid: str):
    """Use FFT amplitude segments: *_amp.npy"""
    return sorted(glob.glob(os.path.join(root, pid, f"{pid}_window*_amp.npy")))

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
                arr = arr[np.newaxis, :]   # [1, L]
            segs.append(torch.from_numpy(arr).float())

        return segs, label, pid


def mil_collate_fn(batch):
    segs, labels, pids = zip(*batch)
    return list(segs), torch.stack(list(labels)), list(pids)

# ---------------------------------------------------------------
# AlexNet1D Backbone
# ---------------------------------------------------------------
class AlexNet1D(nn.Module):
    """
    A 1D version of AlexNet for ECG Fourier amplitude segments.
    Input: [B, 1, L]
    Output: embedding vector (emb_dim)
    """
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

        # Adaptive pool to compress temporal dimension
        self.avgpool = nn.AdaptiveAvgPool1d(6)

        self.fc = nn.Sequential(
            nn.Linear(256 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, emb_dim)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x).squeeze(0)

# ---------------------------------------------------------------
# Attention MIL + Classifier (AlexNet version)
# ---------------------------------------------------------------
class AttentionMIL(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, H):
        A = torch.softmax(self.w(torch.tanh(self.V(H))).squeeze(-1), dim=0)
        Z = torch.sum(A.unsqueeze(-1) * H, dim=0)
        return Z, A


class MILAlexNetClassifier(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128,
                 attn_hidden=128, clf_hidden=64, dropout=0.1):
        super().__init__()
        self.encoder = AlexNet1D(in_ch=in_ch, emb_dim=emb_dim)
        self.pool = AttentionMIL(emb_dim, attn_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, clf_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden, 1)
        )

    def forward(self, segs):
        H = torch.stack([self.encoder(s) for s in segs])
        Z, A = self.pool(H)
        return self.classifier(Z).squeeze(-1), A

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
        return {k: float("nan") for k in ["AUC","ACC","Precision","Recall","F1","Threshold"]}

    auc = roc_auc_score(ys, ps)
    prec, rec, thr = precision_recall_curve(ys, ps)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    i = np.argmax(f1)
    thr_opt = thr[i] if i < len(thr) else 0.5
    preds = (np.array(ps) >= thr_opt).astype(int)

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
    loss_fn = nn.BCEWithLogitsLoss()
    total = 0.0

    for bags, labels, _ in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        loss_batch = 0.0

        for segs, y in zip(bags, labels.to(device)):
            segs = [s.to(device) for s in segs]

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits, _ = model(segs)
                loss = loss_fn(logits, y)

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_batch += loss.item()

        if scaler:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()

        total += loss_batch

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
    save_dir: str = "mil_runs_fft"
    run_name: str = "alexnet1d_mil_cv_fft"
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

    seg_root = os.path.join(cfg.data_root, "fft_amplitude_segments")

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(meta, meta["outcome_label"]), start=1):
        print(f"\n========== Fold {fold}/{cfg.n_splits} ==========")

        ds_tr = ECGMILDataset(meta.iloc[tr_idx], seg_root, cfg.max_segments)
        ds_val = ECGMILDataset(meta.iloc[val_idx], seg_root, cfg.max_segments)

        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, collate_fn=mil_collate_fn)
        dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, collate_fn=mil_collate_fn)

        model = MILAlexNetClassifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision else None

        best_auc = -1
        best_metrics = None
        metrics_log = []

        for epoch in range(1, cfg.epochs + 1):
            print(f"\nFold {fold} | Epoch {epoch}/{cfg.epochs}")

            tr_loss = train_one_epoch(model, dl_tr, optimizer, scaler, device)
            val_metrics = evaluate(model, dl_val, device)

            val_metrics.update({"Epoch": epoch, "TrainLoss": tr_loss})
            metrics_log.append(val_metrics)

            print(f"  TrainLoss={tr_loss:.4f} | Val={val_metrics}")

            if not math.isnan(val_metrics["AUC"]) and val_metrics["AUC"] > best_auc:
                best_auc = val_metrics["AUC"]
                best_metrics = val_metrics.copy()

                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.save_dir, f"{cfg.run_name}_fold{fold}_best.pt")
                )
                print(f"  ↳ New best AUC: {best_auc:.4f}")

        pd.DataFrame(metrics_log).to_csv(
            os.path.join(cfg.save_dir, f"{cfg.run_name}_fold{fold}_metrics.csv"),
            index=False
        )

        if best_metrics:
            with open(
                os.path.join(cfg.save_dir, f"{cfg.run_name}_fold{fold}_best_metrics.json"), "w"
            ) as f:
                json.dump(best_metrics, f, indent=2)

            fold_results.append(best_metrics)

        del model
        torch.cuda.empty_cache()

    summary = pd.DataFrame(fold_results)
    mean = summary.mean(numeric_only=True)
    std = summary.std(numeric_only=True)

    summary.loc["Mean"] = mean
    summary.loc["Std"]  = std

    summary_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_cv_summary.csv")
    summary.to_csv(summary_path)

    print("\n===== CROSS-VALIDATION SUMMARY (FFT AlexNet1D) =====")
    for m in ["AUC","ACC","Precision","Recall","F1"]:
        print(f"{m:>10}: {mean[m]:.4f} ± {std[m]:.4f}")
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
