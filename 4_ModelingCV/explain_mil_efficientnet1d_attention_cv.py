#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_mil_efficientnet1d_attention_cv.py
------------------------------------------
Attention-based explainability for 1D EfficientNet MIL model trained with
5-fold Stratified Cross-Validation.

For each fold:
    - Reconstructs validation split used during training
    - Loads fold's best model weights
    - Runs MIL inference on each validation patient
    - Extracts MIL attention (per segment)
    - Saves per-patient summary CSV + full attention vector (.npy)

Uses **top 5%** of segments per patient as "important" (highest attention).

Usage:
python explain_mil_efficientnet1d_attention_cv.py \
    --save_dir mil_runs \
    --run_name efficientnet1d_mil_cv \
    --explain_dir explain_efficientnet_attention \
    --data_root ../../ECG_data \
    --csv_name segmentation_with_labels.csv \
    --segment_dirname preprocessed_segments
"""

import os
import glob
import argparse
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F


# ===============================================================
# Utilities
# ===============================================================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zpad_pid(pid: int, width: int = 3) -> str:
    return str(pid).zfill(width)


def list_patient_segments(root: str, pid: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, pid, f"{pid}_window*.npy")))


# ===============================================================
# Dataset (same as training)
# ===============================================================

class ECGMILDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame, segment_root: str,
                 max_segments: Optional[int] = None):
        self.meta = meta_df.sort_values("patient_id").reset_index(drop=True)
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


# ===============================================================
# EfficientNet1D + Attention MIL (copied from training script)
# ===============================================================

class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, 1)

    def forward(self, x):
        w = x.mean(-1, keepdim=True)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class MBConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=2, stride=1, kernel_size=7):
        super().__init__()
        mid = in_ch * expand_ratio
        padding = kernel_size // 2

        self.use_res = (in_ch == out_ch and stride == 1)

        self.expand = nn.Sequential(
            nn.Conv1d(in_ch, mid, 1, bias=False),
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
            nn.Conv1d(mid, out_ch, 1, bias=False),
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

        base = int(16 * width_mult)

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, 2, 3, bias=False),
            nn.BatchNorm1d(base),
            nn.SiLU()
        )

        # Same block ordering as training
        self.blocks = nn.Sequential(
            MBConv1D(base, base, stride=1),
            MBConv1D(base, base * 2, stride=2),
            MBConv1D(base * 2, base * 2, stride=1),
            MBConv1D(base * 2, base * 4, stride=2),
            MBConv1D(base * 4, base * 4, stride=1),
            MBConv1D(base * 4, base * 4, stride=1),
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
    def __init__(
        self,
        in_ch=1,
        emb_dim=128,
        attn_hidden=128,
        clf_hidden=64,
        dropout=0.1
    ):
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


# ===============================================================
# Config
# ===============================================================

@dataclass
class ExplainConfig:
    data_root: str = "../../ECG_data"
    csv_name: str = "segmentation_with_labels.csv"
    segment_dirname: str = "preprocessed_segments"

    save_dir: str = "mil_runs"
    run_name: str = "efficientnet1d_mil_cv"
    explain_dir: str = "explain_efficientnet_attention"

    max_segments: Optional[int] = None
    batch_size: int = 1
    num_workers: int = 0

    seed: int = 42
    n_splits: int = 5


# ===============================================================
# Explainability Runner
# ===============================================================

def run_explain(cfg: ExplainConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    meta = pd.read_csv(os.path.join(cfg.data_root, cfg.csv_name))
    seg_root = os.path.join(cfg.data_root, cfg.segment_dirname)

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.seed
    )

    base_out = os.path.join(cfg.save_dir, cfg.explain_dir)
    os.makedirs(base_out, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(meta, meta["outcome_label"]), start=1):
        print(f"\n===== EfficientNet Explainability: Fold {fold}/{cfg.n_splits} =====")

        val_meta = meta.iloc[val_idx]
        ds_val = ECGMILDataset(val_meta, seg_root, max_segments=cfg.max_segments)
        dl_val = DataLoader(
            ds_val,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=mil_collate_fn,
        )

        # Load model weights
        model_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_fold{fold}_best.pt")
        if not os.path.exists(model_path):
            print(f"  Missing model weights: {model_path}")
            continue

        print(f"  Loading model weights: {model_path}")
        model = MILEfficientNetClassifier().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Prepare output folder
        fold_dir = os.path.join(base_out, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        rows = []

        # Run attention extraction
        with torch.no_grad():
            for bags, labels, pids in tqdm(dl_val, desc=f"Fold {fold}"):
                for segs, y, pid in zip(bags, labels, pids):

                    segs = [s.to(device) for s in segs]
                    logits, A = model(segs)
                    prob = torch.sigmoid(logits).item()

                    attn = A.cpu().numpy()
                    nseg = len(attn)

                    # ------- TOP 5% ATTENTION -------
                    k = max(1, int(np.ceil(0.05 * nseg)))
                    idx_sorted = np.argsort(attn)[::-1]
                    topk_idx = idx_sorted[:k]
                    topk_vals = attn[topk_idx]

                    # Save attention vector
                    pid_str = str(pid).zfill(3)
                    attn_path = os.path.join(fold_dir, f"attn_patient_{pid_str}.npy")
                    np.save(attn_path, attn)

                    rows.append({
                        "fold": fold,
                        "patient_id": pid_str,
                        "label": float(y.item()),
                        "prob": prob,
                        "logit": float(logits.item()),
                        "n_segments": nseg,
                        "attn_mean": float(attn.mean()),
                        "attn_std": float(attn.std()),
                        "attn_min": float(attn.min()),
                        "attn_max": float(attn.max()),
                        "topk_count": k,
                        "topk_indices": "|".join(map(str, topk_idx.tolist())),
                        "topk_values": "|".join(f"{v:.6f}" for v in topk_vals.tolist()),
                        "attn_npy_path": attn_path
                    })

        # Save summary CSV
        out_csv = os.path.join(fold_dir, "val_attention_summary.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"  Saved attention summary: {out_csv}")


# ===============================================================
# Main
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="Explain MIL EfficientNet1D attention across CV folds.")

    parser.add_argument("--data_root", default="../../ECG_data")
    parser.add_argument("--csv_name", default="segmentation_with_labels.csv")
    parser.add_argument("--segment_dirname", default="preprocessed_segments")

    parser.add_argument("--save_dir", default="mil_runs")
    parser.add_argument("--run_name", default="efficientnet1d_mil_cv")
    parser.add_argument("--explain_dir", default="explain_efficientnet_attention")

    parser.add_argument("--max_segments", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)

    args = parser.parse_args()

    cfg = ExplainConfig(
        data_root=args.data_root,
        csv_name=args.csv_name,
        segment_dirname=args.segment_dirname,
        save_dir=args.save_dir,
        run_name=args.run_name,
        explain_dir=args.explain_dir,
        max_segments=args.max_segments,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        n_splits=args.n_splits,
    )

    run_explain(cfg)


if __name__ == "__main__":
    main()
