#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_attention_efficientnet1d_cv.py
-----------------------------------
Plot MIL attention curves and fold statistics for EfficientNet1D MIL CV.

Usage: python plot_attention_efficientnet1d_cv.py \
  --save_dir mil_runs \
  --explain_dir explain_efficientnet_attention \
  --fold 1 \
  --patient_id 7
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_patient_attention(summary_csv, patient_id, output_path=None):
    df = pd.read_csv(summary_csv)

    pid_str = str(patient_id)
    if pid_str not in df["patient_id"].astype(str).values:
        pid_str = pid_str.zfill(3)
        if pid_str not in df["patient_id"].astype(str).values:
            raise ValueError(f"Patient {patient_id} not found.")

    row = df[df["patient_id"].astype(str) == pid_str].iloc[0]
    attn_path = row["attn_npy_path"]

    if not os.path.exists(attn_path):
        raise FileNotFoundError(attn_path)

    A = np.load(attn_path)
    x = np.arange(len(A))

    # top indices
    if isinstance(row["topk_indices"], str):
        top_idx = [int(i) for i in row["topk_indices"].split("|") if i.strip()]
    else:
        top_idx = []

    plt.figure(figsize=(12, 4))
    plt.plot(x, A, label="Attention")
    if len(top_idx) > 0:
        plt.scatter(top_idx, A[top_idx], color="red", s=20, label="Top 5%")

    plt.xlabel("Segment index")
    plt.ylabel("Attention weight")
    plt.title(f"Patient {pid_str} - Attention")
    plt.legend()
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(summary_csv),
            f"patient_{pid_str}_attention.png"
        )

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fold_stats(summary_csv, output_dir=None):
    df = pd.read_csv(summary_csv)

    if output_dir is None:
        output_dir = os.path.dirname(summary_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Histogram of attention maxima
    plt.figure(figsize=(6, 4))
    df["attn_max"].hist(bins=30)
    plt.xlabel("Maximum attention")
    plt.ylabel("Count")
    plt.title("Distribution of max attention")
    plt.tight_layout()
    out1 = os.path.join(output_dir, "fold_attn_max_hist.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved: {out1}")

    # Histogram of top-k counts
    if "topk_count" in df.columns:
        plt.figure(figsize=(6, 4))
        df["topk_count"].hist(bins=30)
        plt.xlabel("Top-5% segment count")
        plt.ylabel("Count")
        plt.title("Distribution of top-5% count")
        plt.tight_layout()
        out2 = os.path.join(output_dir, "fold_topk_count_hist.png")
        plt.savefig(out2, dpi=150)
        plt.close()
        print(f"Saved: {out2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="mil_runs")
    parser.add_argument("--explain_dir", default="explain_efficientnet_attention")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--patient_id", default=None)
    parser.add_argument("--plot_fold_stats", action="store_true")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    fold_dir = os.path.join(args.save_dir, args.explain_dir, f"fold{args.fold}")
    summary_csv = os.path.join(fold_dir, "val_attention_summary.csv")

    if not os.path.exists(summary_csv):
        raise FileNotFoundError(summary_csv)

    if args.patient_id is not None:
        out_dir = args.output_dir or fold_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"patient_{args.patient_id}_attention.png")
        plot_patient_attention(summary_csv, args.patient_id, out_path)

    if args.plot_fold_stats:
        out_dir = args.output_dir or fold_dir
        plot_fold_stats(summary_csv, out_dir)


if __name__ == "__main__":
    main()
