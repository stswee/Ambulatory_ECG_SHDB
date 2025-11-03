#!/usr/bin/env python3
"""
create_window_index_metadata.py

Goal:
-----
Generate window_index_metadata.csv for on-the-fly ECG segmentation.

Each row corresponds to one preprocessed ECG (.npz) file and includes:
    patient_id, fs, window_sec, signal_length, n_windows, start_indices

Usage:
------
python create_window_index_metadata.py \
    --input_dir ../../../../local3/sswee/data/shdbaf/physionet.org/files/shdb-af/preprocessed \
    --output_dir ../../../../local3/sswee/data/shdbaf/physionet.org/files/shdb-af \
    --fs 200 \
    --window_sec 30
"""

import os
import csv
import argparse
import numpy as np
from tqdm import tqdm


def generate_window_indices(input_dir, output_dir, fs=200, window_sec=30):
    """Scan preprocessed ECGs and compute 30s window start indices."""
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "window_index_metadata.csv")

    win_len = fs * window_sec
    npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "fs", "window_sec", "signal_length", "n_windows", "start_indices"])

        for fname in tqdm(npz_files, desc="Processing ECGs", unit="file"):
            fpath = os.path.join(input_dir, fname)
            try:
                data = np.load(fpath)
                sig = data["signal"]
                if sig.ndim > 1:
                    sig = sig[0]  # (C, N) → (N,)
                T = len(sig)

                n_windows = T // win_len
                if n_windows == 0:
                    print(f"[!] Skipping {fname}: too short for one {window_sec}s window.")
                    continue

                starts = list(range(0, n_windows * win_len, win_len))
                pid = str(data.get("record_name", os.path.basename(fname).split("_")[0]))
                if isinstance(pid, np.ndarray):
                    pid = pid.item()

                writer.writerow([pid, fs, window_sec, T, n_windows, str(starts)])
            except Exception as e:
                print(f"[!] Error on {fname}: {e}")

    print(f"\nMetadata saved to: {output_csv}")
    print(f"Total processed ECGs: {len(npz_files)}")


def main():
    parser = argparse.ArgumentParser(description="Generate window index metadata for ECGs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to preprocessed ECG folder.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save window_index_metadata.csv.")
    parser.add_argument("--fs", type=int, default=200, help="Sampling frequency (Hz).")
    parser.add_argument("--window_sec", type=int, default=30, help="Window duration (seconds).")
    args = parser.parse_args()

    generate_window_indices(args.input_dir, args.output_dir, fs=args.fs, window_sec=args.window_sec)


if __name__ == "__main__":
    main()
