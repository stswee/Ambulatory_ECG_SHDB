#!/usr/bin/env python3
"""
segment_all_ecgs.py

Goal:
-----
Segment each preprocessed ECG (.npz) into fixed 30-second windows
and save each patient's windows into a single compressed file in
the 'segmented/' folder.

Also generates segmentation_metadata.csv containing:
  patient_id, n_windows, fs, window_sec, signal_length, output_file

Usage:
------
  python segment_all_ecgs.py --input_dir /path/to/preprocessed --fs 200 --window_sec 30

Output:
-------
data/segmented/
  ├── 001_windows.npz
  ├── 002_windows.npz
  ├── ...
data/metadata/segmentation_metadata.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def segment_ecg(npz_path, out_dir, fs=200, window_sec=30):
    """Split ECG into fixed 30-second windows and save as .npz."""
    data = np.load(npz_path)
    signal = data["signal"]

    # Ensure shape (C, T)
    if signal.ndim == 1:
        signal = signal[None, :]
    C, T = signal.shape

    # Compute window parameters
    win_len = fs * window_sec
    n_windows = T // win_len
    if n_windows == 0:
        raise ValueError(f"Signal too short ({T} samples) for {window_sec}s windows.")

    # Trim to multiple of window length
    trimmed = signal[:, :n_windows * win_len]

    # Reshape to (N_windows, C, window_len)
    windows = trimmed.reshape(C, n_windows, win_len).transpose(1, 0, 2)

    # Derive patient ID from filename or metadata
    if "record_name" in data:
        pid = str(data["record_name"].item())
    else:
        pid = os.path.basename(npz_path).split("_")[0]

    # Save segmented file
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{pid}_windows.npz")
    np.savez_compressed(
        save_path,
        windows=windows.astype(np.float32),
        patient_id=pid,
        fs=fs,
        window_sec=window_sec,
        n_windows=n_windows,
    )

    # Return metadata row
    return {
        "patient_id": pid,
        "n_windows": n_windows,
        "fs": fs,
        "window_sec": window_sec,
        "signal_length": T,
        "output_file": save_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Segment all preprocessed ECGs into 30s windows.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed .npz ECG files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save segmented ECGs (default: sibling 'segmented' folder).",
    )
    parser.add_argument("--fs", type=int, default=200, help="Sampling frequency in Hz.")
    parser.add_argument("--window_sec", type=int, default=30, help="Window length in seconds.")
    args = parser.parse_args()

    # Define paths
    input_dir = os.path.abspath(args.input_dir)
    base_dir = os.path.dirname(input_dir)
    out_dir = args.output_dir or os.path.join(base_dir, "segmented")
    meta_dir = os.path.join(base_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, "segmentation_metadata.csv")

    print(f"\nInput directory:  {input_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Metadata file:    {meta_path}\n")

    npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    metadata = []
    for fname in tqdm(npz_files, desc="Segmenting ECGs", unit="file"):
        fpath = os.path.join(input_dir, fname)
        try:
            meta = segment_ecg(fpath, out_dir, fs=args.fs, window_sec=args.window_sec)
            metadata.append(meta)
        except Exception as e:
            print(f"[!] Error on {fname}: {e}")

    # Write metadata CSV
    df = pd.DataFrame(metadata)
    df.to_csv(meta_path, index=False)

    print(f"\n✅ Segmentation complete.")
    print(f"→ Segmented files saved to: {out_dir}")
    print(f"→ Metadata saved to:        {meta_path}\n")


if __name__ == "__main__":
    main()
