#!/usr/bin/env python3
"""
merge_metadata_files.py

Goal:
-----
Merge the three core metadata CSVs:
    1. window_index_metadata.csv
    2. patient_labels.csv
    3. dataset_splits.csv

→ Produces a single master file: segmentation_with_labels.csv

Each row = one patient (bag), containing:
    patient_id, outcome_label, split, fs, window_sec,
    signal_length, n_windows, start_indices

Usage:
------
python merge_metadata_files.py \
  --base_dir ../../../../local3/sswee/data/shdbaf/physionet.org/files/shdb-af
"""

import os
import argparse
import pandas as pd


def merge_metadata(base_dir):
    """Merge window index, labels, and dataset split metadata into one CSV."""
    # Input files
    window_csv = os.path.join(base_dir, "window_index_metadata.csv")
    labels_csv = os.path.join(base_dir, "patient_labels.csv")
    splits_csv = os.path.join(base_dir, "dataset_splits.csv")

    # Output file
    output_csv = os.path.join(base_dir, "segmentation_with_labels.csv")

    # Check existence
    for path in [window_csv, labels_csv, splits_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    # Load
    df_window = pd.read_csv(window_csv)
    df_labels = pd.read_csv(labels_csv)
    df_splits = pd.read_csv(splits_csv)

    # Clean column names
    for df in [df_window, df_labels, df_splits]:
        df.columns = [c.strip().lower() for c in df.columns]

    # Ensure consistent patient_id type
    for df in [df_window, df_labels, df_splits]:
        df["patient_id"] = df["patient_id"].astype(str).str.zfill(3)

    # Merge
    merged = (
        df_window
        .merge(df_labels, on="patient_id", how="inner")
        .merge(df_splits, on="patient_id", how="inner")
    )

    # Reorder columns
    columns = [
        "patient_id", "outcome_label", "split",
        "fs", "window_sec", "signal_length", "n_windows", "start_indices"
    ]
    merged = merged[columns]

    # Save
    merged.to_csv(output_csv, index=False)

    # Summary
    print(f"Saved merged metadata to: {output_csv}")
    print(f"Rows: {len(merged)}")
    print(f"Columns: {list(merged.columns)}")
    print("\nSplit distribution:")
    print(merged["split"].value_counts())
    print("\nClass distribution:")
    print(merged["outcome_label"].value_counts(normalize=True).round(3))

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge all metadata CSVs for ECG MIL training.")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Path to base directory containing the CSV files.")
    args = parser.parse_args()

    merge_metadata(args.base_dir)


if __name__ == "__main__":
    main()
