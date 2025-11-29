#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_attention_alexnet1d_cv.py
------------------------------
Plot MIL attention for AlexNet1D MIL CV model.

Requires outputs from:
    explain_mil_alexnet1d_attention_cv.py

Per fold:
- Reads:  {save_dir}/{explain_dir}/fold{fold}/val_attention_summary.csv
- Optionally plots per-patient attention timeline.
- Optionally plots fold-level statistics (e.g., histogram of attn_max).

Usage:
python plot_attention_alexnet1d_cv.py   --save_dir mil_runs   --explain_dir explain_attention   --fold 1   --patient_id 7
python plot_attention_alexnet1d_cv.py   --save_dir mil_runs   --explain_dir explain_attention   --fold 1   --plot_fold_stats
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_patient_attention(summary_csv, patient_id, output_path=None):
    """
    Plot attention vs segment index for a specific patient.
    Highlights the top 5% segments using the precomputed indices.
    """

    df = pd.read_csv(summary_csv)

    # patient_id stored as string (e.g., "001"); accept either "1" or "001"
    pid_str = str(patient_id)
    if pid_str not in df["patient_id"].astype(str).values:
        # Try zero-padded version
        pid_str = pid_str.zfill(3)
        if pid_str not in df["patient_id"].astype(str).values:
            raise ValueError(f"Patient {patient_id} not found in {summary_csv}")

    row = df[df["patient_id"].astype(str) == pid_str].iloc[0]
    attn_path = row["attn_npy_path"]

    if not os.path.exists(attn_path):
        raise FileNotFoundError(f"Attention file not found: {attn_path}")

    attn = np.load(attn_path)
    nseg = len(attn)

    # Parse top-5% indices and values from CSV
    if isinstance(row["topk_indices"], str) and row["topk_indices"]:
        topk_idx = np.array([int(x) for x in row["topk_indices"].split("|")])
    else:
        topk_idx = np.array([], dtype=int)

    x = np.arange(nseg)

    plt.figure(figsize=(12, 4))
    plt.plot(x, attn, label="Attention", linewidth=1)

    if topk_idx.size > 0:
        plt.scatter(topk_idx, attn[topk_idx], color="red", s=20, label="Top 5% segments")

    plt.xlabel("Segment index")
    plt.ylabel("Attention weight")
    plt.title(f"Patient {pid_str} - Attention over segments")
    plt.legend()
    plt.tight_layout()

    if output_path is None:
        # Default output path next to summary_csv
        base_dir = os.path.dirname(summary_csv)
        output_path = os.path.join(base_dir, f"patient_{pid_str}_attention.png")

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved attention plot for patient {pid_str} to: {output_path}")


def plot_fold_stats(summary_csv, output_dir=None):
    """
    Plot simple fold-level stats:
    - Histogram of attn_max across patients
    - Histogram of topk_count (# segments in top 5%)
    """

    df = pd.read_csv(summary_csv)

    if output_dir is None:
        output_dir = os.path.dirname(summary_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Histogram of attn_max
    plt.figure(figsize=(6, 4))
    df["attn_max"].hist(bins=30)
    plt.xlabel("attn_max")
    plt.ylabel("Count")
    plt.title("Distribution of max attention per patient")
    plt.tight_layout()
    out1 = os.path.join(output_dir, "fold_attn_max_hist.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved fold-level attn_max histogram to: {out1}")

    # Histogram of topk_count (how many segments in top 5%)
    if "topk_count" in df.columns:
        plt.figure(figsize=(6, 4))
        df["topk_count"].hist(bins=30)
        plt.xlabel("topk_count (top 5% segments)")
        plt.ylabel("Count")
        plt.title("Distribution of #segments in top 5% per patient")
        plt.tight_layout()
        out2 = os.path.join(output_dir, "fold_topk_count_hist.png")
        plt.savefig(out2, dpi=150)
        plt.close()
        print(f"Saved fold-level topk_count histogram to: {out2}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot MIL attention for AlexNet1D MIL CV model."
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
        default=None,
        help="Patient ID to plot attention for (e.g., '1' or '001').",
    )
    parser.add_argument(
        "--plot_fold_stats",
        action="store_true",
        help="If set, also plot fold-level histograms.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (defaults to fold directory).",
    )

    args = parser.parse_args()

    fold_dir = os.path.join(args.save_dir, args.explain_dir, f"fold{args.fold}")
    summary_csv = os.path.join(fold_dir, "val_attention_summary.csv")

    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    if args.patient_id is not None:
        # If no custom output_dir given, use fold_dir
        if args.output_dir is None:
            out_dir = fold_dir
        else:
            out_dir = args.output_dir
            os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"patient_{args.patient_id}_attention.png")
        plot_patient_attention(summary_csv, args.patient_id, output_path=out_path)

    if args.plot_fold_stats:
        if args.output_dir is None:
            out_dir = fold_dir
        else:
            out_dir = args.output_dir
        plot_fold_stats(summary_csv, output_dir=out_dir)


if __name__ == "__main__":
    main()
