#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overlay_attention_ecg_efficientnet1d_cv.py
------------------------------------------
Overlay MIL attention on raw ECG segments for EfficientNet1D MIL CV model.

For a given fold and patient:
    - Load attention vector from explain_mil_efficientnet1d_attention_cv.py
    - Compute top 5% highest-attention segments
    - Load corresponding ECG segments (.npy)
    - Plot:
        * Attention vs segment index (top panel)
        * Raw ECG for top-K segments (bottom panels)

Usage:
python overlay_attention_ecg_efficientnet1d_cv.py \
    --data_root ../../ECG_data \
    --segment_dirname preprocessed_segments \
    --save_dir mil_runs \
    --explain_dir explain_efficientnet_attention \
    --fold 1 \
    --patient_id 7 \
    --top_fraction 0.05 \
    --max_plots 6
"""

import os
import glob
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===============================================================
# Utilities
# ===============================================================

def zpad_pid(pid: int, width: int = 3) -> str:
    return str(pid).zfill(width)


def list_patient_segments(segment_root: str, pid: str) -> List[str]:
    pattern = os.path.join(segment_root, pid, f"{pid}_window*.npy")
    files = sorted(glob.glob(pattern))
    return files


def load_patient_row(summary_csv: str, patient_id: str) -> pd.Series:
    df = pd.read_csv(summary_csv)
    pid_str = str(patient_id)

    # Exact match
    if pid_str in df["patient_id"].astype(str).values:
        return df[df["patient_id"].astype(str) == pid_str].iloc[0]

    # Try zero-padding
    pid_str = pid_str.zfill(3)
    if pid_str in df["patient_id"].astype(str).values:
        return df[df["patient_id"].astype(str) == pid_str].iloc[0]

    raise ValueError(f"Patient {patient_id} not found in {summary_csv}")


# ===============================================================
# Core function
# ===============================================================

def overlay_attention_with_ecg(
    data_root: str,
    segment_dirname: str,
    save_dir: str,
    explain_dir: str,
    fold: int,
    patient_id: str,
    top_fraction: float = 0.05,
    max_plots: int = 8,
    output_path: str = None,
):
    """
    Overlay EfficientNet1D attention with ECG segments.
    """

    # Paths
    fold_dir = os.path.join(save_dir, explain_dir, f"fold{fold}")
    summary_csv = os.path.join(fold_dir, "val_attention_summary.csv")

    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    # Get patient row
    row = load_patient_row(summary_csv, patient_id)
    attn_path = row["attn_npy_path"]

    if not os.path.exists(attn_path):
        raise FileNotFoundError(f"Attention .npy not found: {attn_path}")

    # Load attention vector
    attn = np.load(attn_path)
    nseg = len(attn)

    # Compute top-k indices
    k = max(1, int(np.ceil(top_fraction * nseg)))
    idx_sorted = np.argsort(attn)[::-1]
    top_idx = idx_sorted[:k]

    # Load ECG segment files
    pid_str = str(row["patient_id"]).zfill(3)
    segment_root = os.path.join(data_root, segment_dirname)
    seg_files = list_patient_segments(segment_root, pid_str)

    # Safety check for mismatch
    if len(seg_files) != nseg:
        print(
            f"[Warning] Segment count mismatch: {len(seg_files)} files vs {nseg} attention values. "
            f"Using min length."
        )
        nmin = min(len(seg_files), nseg)
        seg_files = seg_files[:nmin]
        attn = attn[:nmin]
        nseg = nmin
        k = max(1, int(np.ceil(top_fraction * nseg)))
        idx_sorted = np.argsort(attn)[::-1]
        top_idx = idx_sorted[:k]

    top_idx_to_plot = top_idx[:max_plots]

    # ===========================================================
    # Plotting
    # ===========================================================

    nrows = 1 + len(top_idx_to_plot)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, figsize=(12, 2.6 * nrows), sharex=False
    )
    if nrows == 1:
        axes = [axes]

    # -------------------------------
    # Panel 1: Attention timeline
    # -------------------------------
    ax_attn = axes[0]
    x = np.arange(nseg)
    ax_attn.plot(x, attn, linewidth=1, label="Attention")
    ax_attn.scatter(
        top_idx_to_plot,
        attn[top_idx_to_plot],
        color="red",
        s=20,
        label=f"Top {top_fraction*100:.1f}%",
    )
    ax_attn.set_ylabel("Attention")
    ax_attn.set_title(
        f"EfficientNet1D | Fold {fold} | Patient {pid_str} | Attention Over Segments"
    )
    ax_attn.legend(loc="upper right")

    # -------------------------------
    # Subsequent panels: Raw ECG
    # -------------------------------
    for i, seg_idx in enumerate(top_idx_to_plot, start=1):
        ax = axes[i]
        seg_file = seg_files[seg_idx]
        ecg = np.load(seg_file)

        # ECG may be shape (L,) or (C, L)
        trace = ecg if ecg.ndim == 1 else ecg[0]

        ax.plot(trace, linewidth=0.9)
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"Segment {seg_idx} | attention={attn[seg_idx]:.4f}"
        )

    axes[-1].set_xlabel("Sample index")
    plt.tight_layout()

    # -------------------------------
    # Output path
    # -------------------------------
    if output_path is None:
        output_path = os.path.join(
            fold_dir,
            f"efficientnet_fold{fold}_patient_{pid_str}_overlay_top{int(top_fraction*100)}pct.png",
        )

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved overlay figure to: {output_path}")


# ===============================================================
# Main CLI
# ===============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Overlay MIL attention on ECG segments for EfficientNet1D MIL CV model."
    )

    parser.add_argument("--data_root", type=str, default="../../ECG_data")
    parser.add_argument("--segment_dirname", type=str, default="preprocessed_segments")

    parser.add_argument("--save_dir", type=str, default="mil_runs")
    parser.add_argument("--explain_dir", type=str, default="explain_efficientnet_attention")

    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--patient_id", type=str, required=True)

    parser.add_argument("--top_fraction", type=float, default=0.05)
    parser.add_argument("--max_plots", type=int, default=8)

    parser.add_argument("--output_path", type=str, default=None)

    args = parser.parse_args()

    overlay_attention_with_ecg(
        data_root=args.data_root,
        segment_dirname=args.segment_dirname,
        save_dir=args.save_dir,
        explain_dir=args.explain_dir,
        fold=args.fold,
        patient_id=args.patient_id,
        top_fraction=args.top_fraction,
        max_plots=args.max_plots,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
