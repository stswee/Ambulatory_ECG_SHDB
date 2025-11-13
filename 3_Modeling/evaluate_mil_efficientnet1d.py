#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_mil_efficientnet1d.py
------------------------------
Evaluate a trained MIL-EfficientNet1D model on the test dataset.

Assumes you already trained and saved:
    mil_runs/efficientnet1d_mil_best.pt

and that your data is organized as:
    ../../ECG_data/segmentation_with_labels.csv
    ../../ECG_data/preprocessed_segments/

Outputs:
    mil_runs/efficientnet1d_mil_test_metrics.json
    mil_runs/efficientnet1d_mil_test_predictions.csv
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve


# ============================================================
# Dataset utilities (identical to train script)
# ============================================================
def zpad_pid(pid: int, width: int = 3):
    return str(pid).zfill(width)

def list_patient_segments(segment_root: str, pid_str: str):
    import glob
    return sorted(glob.glob(os.path.join(segment_root, pid_str, f"{pid_str}_window*.npy")))

class ECGMILDataset(Dataset):
    def __init__(self, csv_path: str, segment_root: str, split: str = "test", max_segments: int = None):
        self.meta = pd.read_csv(csv_path)
        self.meta = self.meta[self.meta["split"].str.lower() == split.lower()].reset_index(drop=True)
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


# ============================================================
# Model components (EfficientNet1D + MIL)
# ============================================================
import torch.nn.functional as F

class MBConv1D(nn.Module):
    """Mobile Inverted Residual Block (1D version)."""
    def __init__(self, in_ch, out_ch, expand_ratio=4, stride=1, se_ratio=0.25, drop_connect=0.0):
        super().__init__()
        hidden_ch = int(in_ch * expand_ratio)
        self.stride = stride
        self.drop_connect = drop_connect

        self.expand = nn.Sequential(
            nn.Conv1d(in_ch, hidden_ch, 1, bias=False),
            nn.BatchNorm1d(hidden_ch),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()

        self.dwconv = nn.Sequential(
            nn.Conv1d(hidden_ch, hidden_ch, 3, stride=stride, padding=1, groups=hidden_ch, bias=False),
            nn.BatchNorm1d(hidden_ch),
            nn.SiLU()
        )

        # Squeeze and Excitation
        se_hidden = max(1, int(in_ch * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_ch, se_hidden, 1),
            nn.SiLU(),
            nn.Conv1d(se_hidden, hidden_ch, 1),
            nn.Sigmoid()
        )

        self.project = nn.Sequential(
            nn.Conv1d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch)
        )

        self.use_res_connect = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.dwconv(out)
        out = out * self.se(out)
        out = self.project(out)
        if self.use_res_connect:
            out = out + identity
        return out


class EfficientNet1D(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128, width_mult=1.0):
        super().__init__()
        b = lambda x: int(x * width_mult)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, b(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(b(32)),
            nn.SiLU()
        )

        self.blocks = nn.Sequential(
            MBConv1D(b(32),  b(16),  expand_ratio=1, stride=1),
            MBConv1D(b(16),  b(24),  expand_ratio=6, stride=2),
            MBConv1D(b(24),  b(24),  expand_ratio=6, stride=1),
            MBConv1D(b(24),  b(40),  expand_ratio=6, stride=2),
            MBConv1D(b(40),  b(40),  expand_ratio=6, stride=1),
            MBConv1D(b(40),  b(80),  expand_ratio=6, stride=2),
            MBConv1D(b(80),  b(80),  expand_ratio=6, stride=1)
        )

        self.head = nn.Sequential(
            nn.Conv1d(b(80), b(1280), 1, bias=False),
            nn.BatchNorm1d(b(1280)),
            nn.SiLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(b(1280), emb_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x).squeeze(-1)
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
    def __init__(self, in_ch=1, emb_dim=128, attn_hidden=128, clf_hidden=64, dropout=0.1):
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


# ============================================================
# Evaluation
# ============================================================
def evaluate(model, loader, device):
    model.eval()
    ys, ps, pids = [], [], []
    with torch.no_grad():
        for segments_bags, labels, patient_ids in tqdm(loader, desc="Testing"):
            for segments, y, pid in zip(segments_bags, labels, patient_ids):
                segments = [s.to(device) for s in segments]
                logits, _ = model(segments)
                p = torch.sigmoid(logits)
                ys.append(y.item())
                ps.append(float(p.item()))
                pids.append(pid)

    precision, recall, thresholds = precision_recall_curve(ys, ps)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    preds = [1 if p >= best_thr else 0 for p in ps]
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    acc = accuracy_score(ys, preds)
    prec = precision_score(ys, preds, zero_division=0)
    rec = recall_score(ys, preds, zero_division=0)

    results = {
        "AUC": auc,
        "ACC": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": best_f1,
        "Optimal_Threshold": best_thr,
        "Num_Patients": len(ys)
    }

    pred_df = pd.DataFrame({
        "patient_id": pids,
        "true_label": ys,
        "pred_prob": ps,
        "pred_label": preds
    })

    return results, pred_df


# ============================================================
# Main
# ============================================================
def main():
    data_root = "../../ECG_data"
    csv_path = os.path.join(data_root, "segmentation_with_labels.csv")
    seg_root = os.path.join(data_root, "preprocessed_segments")
    weights_path = "mil_runs/efficientnet1d_mil_best.pt"
    save_dir = "mil_runs"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_test = ECGMILDataset(csv_path, seg_root, split="test", max_segments=None)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=mil_collate_fn)

    model = MILEfficientNetClassifier().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded model weights from: {weights_path}")

    results, pred_df = evaluate(model, dl_test, device)

    # Save results
    json_path = os.path.join(save_dir, "efficientnet1d_mil_test_metrics.json")
    csv_path = os.path.join(save_dir, "efficientnet1d_mil_test_predictions.csv")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    pred_df.to_csv(csv_path, index=False)

    print("\n=== Test Results ===")
    for k, v in results.items():
        print(f"{k:>15}: {v:.4f}" if isinstance(v, float) else f"{k:>15}: {v}")
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
