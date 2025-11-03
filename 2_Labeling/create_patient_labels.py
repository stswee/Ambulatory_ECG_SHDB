#!/usr/bin/env python3
"""
create_patient_labels.py

Goal:
-----
Read a CSV file with patient-level conditions (CHF, Stroke, Vascular_Diseases)
and generate a binary label column:
    outcome_label = 1 if (CHF == 1 or Stroke == True or Vascular_Diseases == True)
    outcome_label = 0 otherwise

Usage:
------
python create_patient_labels.py --input AdditionalData.csv --output patient_labels.csv
"""

import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate binary patient labels from condition data.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV (original database).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output CSV with labels.")
    args = parser.parse_args()

    # Load the data
    df = pd.read_csv(args.input)

    # Normalize column names (in case of whitespace or case differences)
    df.columns = [c.strip().lower() for c in df.columns]

    # Expected columns (with flexible matching)
    chf_col = [c for c in df.columns if "chf" in c][0]
    stroke_col = [c for c in df.columns if "stroke" in c][0]
    vascular_col = [c for c in df.columns if "vascular" in c][0]
    id_col = df.columns[0]  # assume first column is patient ID

    # Convert booleans to integers where needed
    df[stroke_col] = df[stroke_col].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)
    df[vascular_col] = df[vascular_col].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)
    df[chf_col] = df[chf_col].fillna(0).astype(int)

    # Compute binary label
    df["outcome_label"] = ((df[chf_col] == 1) | (df[stroke_col] == 1) | (df[vascular_col] == 1)).astype(int)

    # Keep only relevant columns
    out_df = df[[id_col, "outcome_label"]].rename(columns={id_col: "patient_id"})

    # Save to CSV
    out_df.to_csv(args.output, index=False)
    print(f"Saved patient labels to: {args.output}")
    print(out_df.head())


if __name__ == "__main__":
    main()
