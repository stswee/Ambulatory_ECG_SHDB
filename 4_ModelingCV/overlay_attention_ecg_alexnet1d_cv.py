#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overlay_attention_ecg_alexnet1d_cv.py
-------------------------------------
Overlay MIL attention on raw ECG segments for AlexNet1D MIL CV model.

For a given fold and patient:
    - Load attention vector from explain_mil_alexnet1d_attention_cv.py
    - Compute top 5% highest-attention segments
    - Load corresponding ECG segments (.npy)
    - Plot:
        * Attention vs segment index (top panel)
        * Raw ECG for top-K segments (bottom panels)

Usage: python overlay_attention_ecg_alexnet1d_cv.py   --data_root ../../ECG_data   --segment_dirname preprocessed_segments   --save_dir mil_runs   --explain_dir explain_attention   --fold 1   --patient_id 7   --top_fraction 0.05   --max_plots 6
"""

import os
import glob
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def zpad_pid(pid: int, width: int = 3) -> str:
    return str(pid).zfill(width)


def list_patient_segments(segment_root: str, pid: str) -> List[str]:
    """List segment files for a given patient (same convention as training)."""
    pattern = os.path.join(segment_root, pid, f"{pid}_window*.npy")
    files = sorted(glob.glob(pattern))
    return files


def load_patient_row(summary_csv: str, patient_id: str) -> pd.Series:
    df = pd.read_csv(summary_csv)
    pid_str = str(patient_id)
    # Try as-is
    if pid_str in df["patient_id"].astype(str).values:
        return df[df["patient_id"].astype(str) == pid_str].iloc[0]
    # Try zero-padded
    pid_str = pid_str.zfill(3)
    if pid_str in df["patient_id"].astype(str).values:
        return df[df["patient_id"].astype(str) == pid_str].iloc[0]
    raise ValueError(f"Patient {patient_id} not found in {summary_csv}")


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
    Create a figure showing:
        - Attention vs segment index
        - Raw ECG for top attention segments
    """

    # Paths
    fold_dir = os.path.join(save_dir, explain_dir, f"fold{fold}")
    summary_csv = os.path.join(fold_dir, "val_attention_summary.csv")
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    row = load_patient_row(summary_csv, patient_id)
    attn_path = row["attn_npy_path"]
    if not os.path.exists(attn_path):
        raise FileNotFoundError(f"Attention .npy not found: {attn_path}")

    attn = np.load(attn_path)
    nseg = len(attn)

    # Compute top-fraction indices
    k = max(1, int(np.ceil(top_fraction * nseg)))
    idx_sorted = np.argsort(attn)[::-1]
    top_idx = idx_sorted[:k]

    # ECG segment files
    pid_str = str(row["patient_id"]).zfill(3)
    segment_root = os.path.join(data_root, segment_dirname)
    seg_files = list_patient_segments(segment_root, pid_str)

    if len(seg_files) != nseg:
        print(
            f"[Warning] Number of segment files ({len(seg_files)}) "
            f"!= length of attention vector ({nseg}). "
            "Will match by min length."
        )
        nmin = min(len(seg_files), nseg)
        seg_files = seg_files[:nmin]
        attn = attn[:nmin]
        nseg = nmin
        # recompute top idx under truncated length
        k = max(1, int(np.ceil(top_fraction * nseg)))
        idx_sorted = np.argsort(attn)[::-1]
        top_idx = idx_sorted[:k]

    # Limit number of segments to plot
    top_idx_to_plot = top_idx[: max_plots]

    # Prepare figure:
    #  - 1 row for attention
    #  - K rows for ECG traces (K <= max_plots)
    nrows = 1 + len(top_idx_to_plot)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, figsize=(12, 2.5 * nrows), sharex=False
    )

    if nrows == 1:
        axes = [axes]

    # --------------- Top panel: attention vs segment index
    ax_attn = axes[0]
    x = np.arange(nseg)
    ax_attn.plot(x, attn, linewidth=1, label="Attention")
    ax_attn.scatter(top_idx_to_plot, attn[top_idx_to_plot], color="red", s=20,
                    label=f"Top {top_fraction*100:.1f}% (up to {max_plots})")
    ax_attn.set_ylabel("Attention")
    ax_attn.set_xlabel("Segment index")
    ax_attn.set_title(f"Fold {fold} - Patient {pid_str} - Attention over segments")
    ax_attn.legend(loc="upper right")

    # --------------- Lower panels: ECG traces for top segments
    for i, seg_idx in enumerate(top_idx_to_plot, start=1):
        ax = axes[i]
        seg_file = seg_files[seg_idx]
        ecg = np.load(seg_file)

        # ecg shape: [C, L] or [1, L]; plot first channel
        if ecg.ndim == 1:
            trace = ecg
        else:
            trace = ecg[0]

        ax.plot(trace, linewidth=0.8)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Top segment #{i}: index={seg_idx}, attention={attn[seg_idx]:.4f}")

    axes[-1].set_xlabel("Sample index")

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(
            fold_dir,
            f"fold{fold}_patient_{pid_str}_overlay_top{int(top_fraction*100)}pct.png",
        )

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved overlay figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay MIL attention on ECG segments for AlexNet1D MIL CV model."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../ECG_data",
        help="Root directory of ECG data (contains segmentation_with_labels.csv, preprocessed_segments/).",
    )
    parser.add_argument(
        "--segment_dirname",
        type=str,
        default="preprocessed_segments",
        help="Subdirectory under data_root containing segment .npy files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="mil_runs",
        help="Directory where explainability outputs were saved.",
    )
    parser.add_argument(
        "--explain_dir",
        type=str,
        default="explain_attention",
        help="Subdirectory under save_dir with explainability outputs.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold number (1-based).",
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        required=True,
        help="Patient ID to visualize (e.g., '1' or '001').",
    )
    parser.add_argument(
        "--top_fraction",
        type=float,
        default=0.05,
        help="Fraction of highest-attention segments to consider (e.g., 0.05 for top 5%).",
    )
    parser.add_argument(
        "--max_plots",
        type=int,
        default=8,
        help="Maximum number of top segments to show ECG traces for.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Custom output path for the figure (PNG).",
    )

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
