#!/usr/bin/env python3
"""
create_dataset_splits.py

Goal:
-----
Split patients into train/val/test sets with stratification by outcome_label.

Input:
    patient_labels.csv  →  (patient_id, outcome_label)

Output:
    dataset_splits.csv  →  (patient_id, split)

Usage:
------
python create_dataset_splits.py \
  --input patient_labels.csv \
  --output dataset_splits.csv \
  --train_frac 0.7 --val_frac 0.15 --test_frac 0.15 \
  --seed 42
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test dataset splits for MIL.")
    parser.add_argument("--input", type=str, required=True, help="Path to patient_labels.csv")
    parser.add_argument("--output", type=str, required=True, help="Path to save dataset_splits.csv")
    parser.add_argument("--train_frac", type=float, default=0.7, help="Fraction of data for training")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Fraction of data for validation")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Fraction of data for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load labels
    df = pd.read_csv(args.input)
    if not {"patient_id", "outcome_label"}.issubset(df.columns):
        raise ValueError("Input CSV must contain columns: patient_id, outcome_label")

    # Verify fractions
    total = args.train_frac + args.val_frac + args.test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1. Got {total:.3f}")

    # Stratified splits
    train_df, temp_df = train_test_split(
        df, stratify=df["outcome_label"], test_size=(1 - args.train_frac), random_state=args.seed
    )
    val_rel = args.val_frac / (args.val_frac + args.test_frac)
    val_df, test_df = train_test_split(
        temp_df, stratify=temp_df["outcome_label"], test_size=(1 - val_rel), random_state=args.seed
    )

    # Assign splits
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Concatenate and save
    split_df = pd.concat([train_df, val_df, test_df])[["patient_id", "split"]].sort_values("patient_id")
    split_df.to_csv(args.output, index=False)

    # Summary
    print(f"Saved dataset splits to: {args.output}")
    print("\nSplit summary:")
    for split, subdf in split_df.groupby("split"):
        n_pos = (df.set_index("patient_id").loc[subdf["patient_id"], "outcome_label"] == 1).sum()
        n_total = len(subdf)
        print(f"  {split:>5}: {n_total:>4} patients  ({n_pos} positive, {100*n_pos/n_total:.1f}% pos)")


if __name__ == "__main__":
    main()
