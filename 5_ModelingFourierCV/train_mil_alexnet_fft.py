#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mil_alexnet1d_cv_fourier.py
---------------------------------
Multiple-Instance Learning (MIL) using a 1D AlexNet encoder for
Fourier-transformed ECG segments.

Input folders:
    ../../ECG_data/segmentation_with_labels.csv
    ../../ECG_data/fft_amplitude_segments/{pid}/{pid}_windowXXXX_amp.npy
"""

import os, glob, math, json, argparse
from dataclasses import dataclass
from typing import Optional
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

# =====================================================================
# Utilities
# =====================================================================
def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def zpad_pid(pid: int, width: int = 3):
    return str(pid).zfill(width)

def list_patient_segments(root: str, pid: str):
    return sorted(glob.glob(os.path.join(root, pid, f"{pid}_window*_amp.npy")))

# =====================================================================
# Dataset (MIL)
# =====================================================================
class ECGMILDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame, segment_root: str, max_segments: Optional[int] = None):
        self.meta = meta_df.sort_values(by="patient_id").reset_index(drop=True)
        self.segment_root = segment_root
        self.max_segments = max_segments

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        pid = zpad_pid(int(row["patient_id"]))
        label = torch.tensor(float(row["outcome_label"]), dtype=torch.float32)

        seg_files = list_patient_segments(self.segment_root, pid)

        if self.max_segments and len(seg_files) > self.max_segments:
            idxs = np.linspace(0, len(seg_files)-1, self.max_segments, dtype=int)
            seg_files = [seg_files[i] for i in idxs]

        segments = []
        for f in seg_files:
            arr = np.load(f)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]  # [1, L]
            segments.append(torch.from_numpy(arr).float())

        return segments, label, pid

def mil_collate_fn(batch):
    segs, ys, pids = zip(*batch)
    return list(segs), torch.stack(list(ys)), list(pids)

# =====================================================================
# AlexNet-1D Encoder
# =====================================================================
class AlexNet1D(nn.Module):
    """
    Classic AlexNet architecture adapted for 1D signals.
    Fourier input: [1, L]
    """
    def __init__(self, in_channels=1, emb_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=4, padding=2),
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
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        # AlexNet FC layers adapted to your emb_dim
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(512, emb_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        if x.ndim == 2:  # [C, L]
            x = x.unsqueeze(0)

        x = self.features(x)           # [1, 256, L']
        x = self.pool(x).squeeze(-1)   # [1, 256]
        x = self.fc(x).squeeze(0)      # [emb_dim]
        return x

# =====================================================================
# MIL Attention + Classifier
# =====================================================================
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
    def __init__(self, in_ch=1, emb_dim=128, attn_hidden=128, clf_hidden=64, dropout=0.1):
        super().__init__()
        self.encoder = AlexNet1D(in_channels=in_ch, emb_dim=emb_dim)
        self.pool = AttentionMIL(emb_dim, attn_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, clf_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden, 1)
        )

    def forward(self, segments_list):
        feats = [self.encoder(seg) for seg in segments_list]  # list of embeddings
        H = torch.stack(feats)
        z, A = self.pool(H)
        return self.classifier(z).squeeze(-1), A

# =====================================================================
# Training / Evaluation
# =====================================================================
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
    total_loss = 0.0

    for bags, labels, _ in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        batch_loss = 0.0

        for segs, y in zip(bags, labels.to(device)):
            segs = [s.to(device) for s in segs]

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits, _ = model(segs)
                loss = crit(logits, y)

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            batch_loss += loss.item()

        if scaler:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()

        total_loss += batch_loss

    return total_loss / len(loader)

# =====================================================================
# Config
# =====================================================================
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
    save_dir: str = "mil_runs_alexnet_fourier"
    run_name: str = "alexnet1d_mil_cv_fourier"
    seed: int = 42
    n_splits: int = 5

# =====================================================================
# Main
# =====================================================================
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

    # ------------------------
    # Cross-validation loops
    # ------------------------
    for fold, (tr_idx, val_idx) in enumerate(skf.split(meta, meta["outcome_label"]), 1):
        print(f"\n========== Fold {fold}/{cfg.n_splits} ==========")

        ds_tr = ECGMILDataset(meta.iloc[tr_idx], seg_root, cfg.max_segments)
        ds_val = ECGMILDataset(meta.iloc[val_idx], seg_root, cfg.max_segments)

        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, collate_fn=mil_collate_fn)
        dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=mil_collate_fn)

        model = MILAlexNetClassifier(in_ch=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision else None

        metrics_log = []
        best_auc = -1
        best_metrics = None

        for epoch in range(1, cfg.epochs + 1):
            print(f"\nFold {fold} | Epoch {epoch}/{cfg.epochs}")

            tr_loss = train_one_epoch(model, dl_tr, optimizer, scaler, device)
            val_metrics = evaluate(model, dl_val, device)
            val_metrics.update({"TrainLoss": tr_loss, "Epoch": epoch})
            metrics_log.append(val_metrics)

            print(f"  TrainLoss={tr_loss:.4f} | Val={val_metrics}")

            if not math.isnan(val_metrics["AUC"]) and val_metrics["AUC"] > best_auc:
                best_auc = val_metrics["AUC"]
                best_metrics = val_metrics.copy()
                torch.save(model.state_dict(),
                           os.path.join(cfg.save_dir, f"{cfg.run_name}_fold{fold}_best.pt"))
                print(f"  ↳ New best AUC = {best_auc:.4f}")

        pd.DataFrame(metrics_log).to_csv(
            os.path.join(cfg.save_dir, f"{cfg.run_name}_fold{fold}_metrics.csv"), index=False
        )

        if best_metrics:
            with open(os.path.join(cfg.save_dir, f"{cfg.run_name}_fold{fold}_best_metrics.json"), "w") as f:
                json.dump(best_metrics, f, indent=2)
            fold_results.append(best_metrics)

        del model
        torch.cuda.empty_cache()

    # ------------------------
    # CV summary
    # ------------------------
    summary = pd.DataFrame(fold_results)
    mean, std = summary.mean(numeric_only=True), summary.std(numeric_only=True)
    summary.loc["Mean"], summary.loc["Std"] = mean, std

    summary_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_cv_summary.csv")
    summary.to_csv(summary_path)

    print("\n===== Cross-Validation Summary (AlexNet-1D + FFT) =====")
    for m in ["AUC", "ACC", "Precision", "Recall", "F1"]:
        print(f"{m:>10}: {mean[m]:.4f} ± {std[m]:.4f}")
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
