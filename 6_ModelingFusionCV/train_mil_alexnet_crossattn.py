#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mil_alexnet1d_cv_crossattn.py
-----------------------------------
Multiple-Instance Learning (MIL) with 1D AlexNet encoders and
TIME → FREQUENCY Cross-Attention fusion.

TIME-domain segments:
    ../../ECG_data/preprocessed_segments/{pid}/{pid}_windowXXXX.npy
FREQ (FFT amplitude):
    ../../ECG_data/fft_amplitude_segments/{pid}/{pid}_windowXXXX_amp.npy
"""

import os, glob, math, json, argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    precision_recall_curve
)

# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def zpad_pid(pid, width=3):
    return str(pid).zfill(width)

def list_patient_segments(root, pid):
    return sorted(glob.glob(os.path.join(root, pid, f"{pid}_window*.npy")))

# ---------------------------------------------------------------
# Dataset (TIME + FREQ)
# ---------------------------------------------------------------
class ECGMILDataset(Dataset):
    def __init__(self, meta_df, time_root, freq_root, max_segments=None):
        self.meta = meta_df.sort_values("patient_id").reset_index(drop=True)
        self.time_root = time_root
        self.freq_root = freq_root
        self.max_segments = max_segments

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        pid = zpad_pid(int(row["patient_id"]))
        label = torch.tensor(float(row["outcome_label"]), dtype=torch.float32)

        time_files = list_patient_segments(self.time_root, pid)
        if self.max_segments and len(time_files) > self.max_segments:
            time_files = time_files[:self.max_segments]

        time_segs, freq_segs = [], []

        for tf in time_files:
            # load TIME
            arr_t = np.load(tf)
            if arr_t.ndim == 1: arr_t = arr_t[np.newaxis, :]
            time_segs.append(torch.from_numpy(arr_t).float())

            # corresponding FREQ file
            base = os.path.basename(tf).replace(".npy", "")
            ff = os.path.join(self.freq_root, pid, f"{base}_amp.npy")

            if not os.path.exists(ff):
                raise FileNotFoundError(f"Missing freq file: {ff}")

            arr_f = np.load(ff)
            if arr_f.ndim == 1: arr_f = arr_f[np.newaxis, :]
            freq_segs.append(torch.from_numpy(arr_f).float())

        return time_segs, freq_segs, label, pid

def mil_collate_fn(batch):
    t, f, y, p = zip(*batch)
    return list(t), list(f), torch.stack(y), list(p)

# ---------------------------------------------------------------
# AlexNet1D Encoders (TIME / FREQ)
# ---------------------------------------------------------------
class AlexNet1D(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 64, 11, 4, 2), nn.ReLU(), nn.MaxPool1d(3,2),
            nn.Conv1d(64,192,5,1,2), nn.ReLU(), nn.MaxPool1d(3,2),
            nn.Conv1d(192,384,3,1,1), nn.ReLU(),
            nn.Conv1d(384,256,3,1,1), nn.ReLU(),
            nn.Conv1d(256,256,3,1,1), nn.ReLU(),
            nn.MaxPool1d(3,2)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(256, emb_dim)

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(0)
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(0)

# ---------------------------------------------------------------
# Cross-Attention Fusion: TIME → FREQ
# ---------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """
    Asymmetric fusion:
    Ht = time embeddings
    Hf = freq embeddings

    alpha = sigmoid( <Wq Ht, Wk Hf> / sqrt(d) )
    H_fused = Ht + alpha * Wv(Hf)
    """
    def __init__(self, d):
        super().__init__()
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.scale = math.sqrt(d)

    def forward(self, Ht, Hf):
        Q = self.Wq(Ht)
        K = self.Wk(Hf)
        V = self.Wv(Hf)

        scores = (Q*K).sum(-1) / (self.scale + 1e-8)
        alpha  = torch.sigmoid(scores)
        alphaE = alpha.unsqueeze(-1)

        Hfused = Ht + alphaE * V
        return Hfused, alpha

# ---------------------------------------------------------------
# MIL Pool + Classifier
# ---------------------------------------------------------------
class AttentionMIL(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.V = nn.Linear(d, hidden)
        self.w = nn.Linear(hidden,1)

    def forward(self, H):
        A = torch.softmax(self.w(torch.tanh(self.V(H))).squeeze(-1), dim=0)
        z = (A.unsqueeze(-1)*H).sum(0)
        return z, A

class MILAlexNetCrossAttn(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128):
        super().__init__()
        self.enc_t = AlexNet1D(in_ch=in_ch, emb_dim=emb_dim)
        self.enc_f = AlexNet1D(in_ch=in_ch, emb_dim=emb_dim)
        self.fusion = CrossAttentionFusion(emb_dim)
        self.pool = AttentionMIL(emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim,64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64,1)
        )

    def forward(self, ts, fs):
        Ht = torch.stack([self.enc_t(x) for x in ts])
        Hf = torch.stack([self.enc_f(x) for x in fs])

        Hfused, alpha_mod = self.fusion(Ht, Hf)  # [N,d], [N]

        z, Aseg = self.pool(Hfused)
        logit = self.classifier(z).squeeze(-1)
        return logit, (Aseg, alpha_mod)

# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for ts_batch, fs_batch, labels, _ in tqdm(loader, desc="Validating", leave=False):
            for ts, fs, y in zip(ts_batch, fs_batch, labels):
                ts = [x.to(device) for x in ts]
                fs = [x.to(device) for x in fs]
                logit, _ = model(ts, fs)
                ps.append(torch.sigmoid(logit).item())
                ys.append(y.item())

    if len(set(ys)) < 2:
        return {k: float("nan") for k in ["AUC","ACC","Precision","Recall","F1","Threshold"]}

    auc = roc_auc_score(ys,ps)
    prec,rec,thr = precision_recall_curve(ys,ps)
    f1 = 2*prec*rec/(prec+rec+1e-8)
    i  = np.argmax(f1)
    thr_opt = thr[i] if i < len(thr) else 0.5

    preds = [1 if p>=thr_opt else 0 for p in ps]

    return dict(
        AUC=auc,
        ACC=accuracy_score(ys,preds),
        Precision=precision_score(ys,preds,zero_division=0),
        Recall=recall_score(ys,preds,zero_division=0),
        F1=f1[i],
        Threshold=thr_opt
    )

# ---------------------------------------------------------------
# Train One Epoch
# ---------------------------------------------------------------
def train_one_epoch(model, loader, opt, scaler, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total = 0.

    for ts_batch, fs_batch, labels, _ in tqdm(loader, desc="Training", leave=False):
        opt.zero_grad()
        batch_loss = 0.

        for ts, fs, y in zip(ts_batch, fs_batch, labels.to(device)):
            ts = [x.to(device) for x in ts]
            fs = [x.to(device) for x in fs]

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logit, _ = model(ts, fs)
                loss = loss_fn(logit, y)

            if scaler: scaler.scale(loss).backward()
            else: loss.backward()
            batch_loss += loss.item()

        if scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()

        total += batch_loss

    return total / len(loader)

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
@dataclass
class TrainConfig:
    data_root: str = "../../ECG_data"
    csv_name: str  = "segmentation_with_labels.csv"
    time_dir: str  = "preprocessed_segments"
    freq_dir: str  = "fft_amplitude_segments"

    max_segments: Optional[int] = None
    epochs: int = 10
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-2
    mixed_precision: bool = True

    num_workers: int = 0
    save_dir: str = "mil_runs"
    run_name: str = "alexnet1d_crossattn_cv"
    seed: int = 42
    n_splits: int = 5

# ---------------------------------------------------------------
# MAIN
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
    time_root = os.path.join(cfg.data_root, cfg.time_dir)
    freq_root = os.path.join(cfg.data_root, cfg.freq_dir)

    skf = StratifiedKFold(cfg.n_splits, shuffle=True, random_state=cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)
    fold_results = []

    for fold,(tr,val) in enumerate(skf.split(meta, meta["outcome_label"]),1):
        print(f"\n========== Fold {fold}/{cfg.n_splits} ==========")

        ds_tr  = ECGMILDataset(meta.iloc[tr], time_root, freq_root, cfg.max_segments)
        ds_val = ECGMILDataset(meta.iloc[val], time_root, freq_root, cfg.max_segments)

        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, collate_fn=mil_collate_fn)
        dl_val= DataLoader(ds_val,batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, collate_fn=mil_collate_fn)

        model = MILAlexNetCrossAttn().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision else None

        best_auc, best_metrics = -1, None
        logs = []

        for epoch in range(1, cfg.epochs+1):
            print(f"\nFold {fold} | Epoch {epoch}/{cfg.epochs}")
            tr_loss = train_one_epoch(model, dl_tr, opt, scaler, device)
            val_m   = evaluate(model, dl_val, device)
            val_m.update({"TrainLoss": tr_loss, "Epoch": epoch})
            logs.append(val_m)
            print("  TrainLoss={:.4f} | Val={}".format(tr_loss, val_m))

            if not math.isnan(val_m["AUC"]) and val_m["AUC"] > best_auc:
                best_auc = val_m["AUC"]
                best_metrics = val_m.copy()
                torch.save(model.state_dict(),
                           os.path.join(cfg.save_dir,
                                        f"{cfg.run_name}_fold{fold}_best.pt"))
                print("  ↳ New best AUC:", best_auc)

        # save fold logs
        pd.DataFrame(logs).to_csv(
            os.path.join(cfg.save_dir, f"{cfg.run_name}_fold{fold}_metrics.csv"),
            index=False
        )
        with open(os.path.join(cfg.save_dir,
                               f"{cfg.run_name}_fold{fold}_best_metrics.json"),"w") as f:
            json.dump(best_metrics,f,indent=2)

        fold_results.append(best_metrics)
        del model; torch.cuda.empty_cache()

    # summary
    summary = pd.DataFrame(fold_results)
    mean = summary.mean(numeric_only=True)
    std  = summary.std(numeric_only=True)
    summary.loc["Mean"], summary.loc["Std"] = mean, std

    outcsv = os.path.join(cfg.save_dir, f"{cfg.run_name}_summary.csv")
    summary.to_csv(outcsv)

    print("\n===== CROSS-VALIDATION SUMMARY =====")
    for k in ["AUC","ACC","Precision","Recall","F1"]:
        print(f"{k:>10}: {mean[k]:.4f} ± {std[k]:.4f}")
    print("Saved summary to:", outcsv)


if __name__ == "__main__":
    main()
