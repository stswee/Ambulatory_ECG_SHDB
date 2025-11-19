#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mil_resnet1d_cv_hmil.py
-----------------------------
Hierarchical Multiple-Instance Learning (HMIL) with 1D ResNet encoder using
5-fold Stratified Cross-Validation for patient-level ECG classification.

Each fold:
- Trains on 4/5 of patients, validates on 1/5
- Saves best model weights (by AUC)
- Logs metrics per fold
- Saves global summary (mean ± std)

ECG data structure:
    ../../ECG_data/segmentation_with_labels.csv
    ../../ECG_data/preprocessed_segments/

HMIL details:
- Consecutive ECG windows are encoded with ResNet1D
- Embeddings are grouped into contiguous temporal blocks
- A Conv1d-based group aggregator merges blocks hierarchically
- Final patient representation is fed to a classifier head
"""

import os, glob, math, json, argparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # mask all GPUs except GPU 1
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

def zpad_pid(pid: int, width: int = 3) -> str:
    return str(pid).zfill(width)

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

        # Load ALL windows for this patient in temporal order
        files = list_patient_segments(self.segment_root, pid)

        # If max_segments is specified, use FIRST N consecutive windows
        if self.max_segments is not None and len(files) > self.max_segments:
            files = files[:self.max_segments]

        segs = []
        for f in files:
            arr = np.load(f)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]  # (1, L)
            segs.append(torch.from_numpy(arr).float())  # (C, L)

        return segs, label, pid

def mil_collate_fn(batch):
    segs, ys, pids = zip(*batch)
    return list(segs), torch.stack(list(ys)), list(pids)

# ---------------------------------------------------------------
# ResNet1D Encoder
# ---------------------------------------------------------------
class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7,
                               stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = out + identity
        return F.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, in_ch=1, base=64, emb_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base, base, blocks=2, stride=1)
        self.layer2 = self._make_layer(base, base * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base * 2, base * 4, blocks=2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base * 4, emb_dim)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (C, L) or (B, C, L); we expect segments as (C, L) per bag.
        Returns: embedding of shape (emb_dim,) for a single segment.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)   # (1, C, L)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).squeeze(-1)  # (B, C)
        x = self.fc(x)               # (B, emb_dim)
        if x.shape[0] == 1:
            return x.squeeze(0)      # (emb_dim,)
        return x

# ---------------------------------------------------------------
# HMIL: Hierarchical Aggregation Modules
# ---------------------------------------------------------------
class ConvGroupAggregator(nn.Module):
    """
    Aggregates a contiguous group of embeddings of shape (G, D)
    into a single embedding (D,) using a small Conv1d stack
    over the temporal (group) dimension.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.conv2 = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(emb_dim)

    def forward(self, group: torch.Tensor) -> torch.Tensor:
        """
        group: (G, D)
        """
        # Treat D as channels and G as sequence length
        x = group.transpose(0, 1).unsqueeze(0)  # (1, D, G)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # Global average over temporal dimension -> (1, D)
        x = x.mean(dim=-1).squeeze(0)           # (D,)
        return x


class HierarchicalAggregator(nn.Module):
    """
    Hierarchical MIL aggregator:
    - Takes a sequence of embeddings H: (T, D) (T = #segments)
    - Groups consecutive embeddings into blocks of size group_size
    - Applies ConvGroupAggregator at each level
    - Repeats for num_levels or until sequence is short
    - Final representation is mean-pooled across remaining embeddings
    """
    def __init__(self, emb_dim: int, group_size: int = 16, num_levels: int = 2):
        super().__init__()
        self.emb_dim = emb_dim
        self.group_size = group_size
        self.num_levels = num_levels
        self.group_agg = ConvGroupAggregator(emb_dim)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: (T, D)
        Returns: z: (D,) patient-level embedding
        """
        if H.ndim != 2:
            raise ValueError(f"HierarchicalAggregator expects (T, D), got {H.shape}")

        level_H = H
        for _ in range(self.num_levels):
            T = level_H.size(0)
            if T <= self.group_size:
                break

            # Pad to multiple of group_size by repeating last element
            n_groups = math.ceil(T / self.group_size)
            pad_len = n_groups * self.group_size - T
            if pad_len > 0:
                pad = level_H[-1:].repeat(pad_len, 1)  # (pad_len, D)
                level_H = torch.cat([level_H, pad], dim=0)  # (n_groups*group_size, D)

            # Reshape into (n_groups, group_size, D)
            H_groups = level_H.view(n_groups, self.group_size, self.emb_dim)

            # Aggregate each group independently
            group_embs = []
            for g in range(n_groups):
                group_embs.append(self.group_agg(H_groups[g]))  # (D,)

            level_H = torch.stack(group_embs, dim=0)  # (n_groups, D)

        # Final pooling: mean across remaining temporal positions
        if level_H.size(0) == 1:
            z = level_H[0]
        else:
            z = level_H.mean(dim=0)  # (D,)
        return z

# ---------------------------------------------------------------
# HMIL ResNet Classifier
# ---------------------------------------------------------------
class HMILResNetClassifier(nn.Module):
    """
    ResNet1D encoder + hierarchical MIL aggregator + classifier.

    - encoder: ResNet1D → per-window embeddings (emb_dim)
    - h_agg: HierarchicalAggregator → patient embedding
    - classifier: MLP → scalar logit
    """
    def __init__(self,
                 in_ch: int = 1,
                 emb_dim: int = 128,
                 clf_hidden: int = 64,
                 dropout: float = 0.1,
                 hmil_group_size: int = 16,
                 hmil_num_levels: int = 2):
        super().__init__()
        self.encoder = ResNet1D(in_ch=in_ch, emb_dim=emb_dim)
        self.h_agg = HierarchicalAggregator(
            emb_dim=emb_dim,
            group_size=hmil_group_size,
            num_levels=hmil_num_levels
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, clf_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden, 1)
        )

    def forward(self, segments_list):
        """
        segments_list: list of tensors, each (C, L) for one patient.
        Returns:
            logits: scalar tensor
            aux: placeholder (None) for compatibility with older code
        """
        # Encode each 30s window
        feats = [self.encoder(seg) for seg in segments_list]  # list of (emb_dim,)
        H = torch.stack(feats, dim=0)                         # (T, emb_dim)

        # Hierarchical aggregation over consecutive windows
        z = self.h_agg(H)                                     # (emb_dim,)

        # Patient-level prediction
        logits = self.classifier(z).squeeze(-1)               # scalar
        return logits, None

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
        return {m: float("nan")
                for m in ["AUC", "ACC", "Precision", "Recall", "F1", "Threshold"]}

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
    total = 0.0

    for bags, labels, _ in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        loss_sum = 0.0

        for segs, y in zip(bags, labels.to(device)):
            segs = [s.to(device) for s in segs]

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits, _ = model(segs)
                loss = crit(logits, y)

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
    run_name: str = "resnet1d_hmil_cv"
    seed: int = 42
    n_splits: int = 5
    hmil_group_size: int = 16
    hmil_num_levels: int = 2

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
    parser.add_argument("--hmil_group_size", type=int, default=16,
                        help="Number of consecutive windows per group at each HMIL level")
    parser.add_argument("--hmil_num_levels", type=int, default=2,
                        help="Number of hierarchical aggregation levels")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_segments=args.max_segments,
        mixed_precision=args.mixed_precision,
        hmil_group_size=args.hmil_group_size,
        hmil_num_levels=args.hmil_num_levels
    )

    set_seed(cfg.seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")   # because GPU 1 becomes cuda:0 after masking
    print("----- GPU DEBUG INFO -----")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
    print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("---------------------------")

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

        model = HMILResNetClassifier(
            in_ch=1,
            emb_dim=128,
            clf_hidden=64,
            dropout=0.1,
            hmil_group_size=cfg.hmil_group_size,
            hmil_num_levels=cfg.hmil_num_levels
        ).to(device)

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

    # -----------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------
    summary = pd.DataFrame(fold_results)
    mean = summary.mean(numeric_only=True)
    std = summary.std(numeric_only=True)
    summary.loc["Mean"] = mean
    summary.loc["Std"] = std

    csv_path = os.path.join(cfg.save_dir, f"{cfg.run_name}_cv_summary.csv")
    summary.to_csv(csv_path)

    print("\n===== Cross-Validation Summary =====")
    for m in ["AUC", "ACC", "Precision", "Recall", "F1"]:
        print(f"{m:>10}: {mean[m]:.4f} ± {std[m]:.4f}")
    print("Summary CSV saved to:", csv_path)


if __name__ == "__main__":
    main()
