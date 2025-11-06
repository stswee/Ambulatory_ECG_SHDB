#!/usr/bin/env python3
"""
segment_all_patients_by_start_indices.py

Goal:
-----
Segment all preprocessed ECGs into 30-second windows
using the start indices provided in window_index_metadata.csv.

CSV format:
--------------------------------------------------
| patient_id | fs | window_sec | signal_length | n_windows | start_indices |
--------------------------------------------------
start_indices = "[0, 6000, 12000, ...]"
"""

import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
BASE_PATH = "../../../../local3/sswee/data/shdbaf/physionet.org/files/shdb-af"
PREPROCESSED_DIR = os.path.join(BASE_PATH, "preprocessed")
SEGMENTS_DIR = os.path.join(BASE_PATH, "preprocessed_segments")
CSV_PATH = os.path.join(BASE_PATH, "window_index_metadata.csv")

FS = 200
WINDOW_SEC = 30
WINDOW_LEN = FS * WINDOW_SEC  # 6000 samples

# --- Load metadata ---
meta = pd.read_csv(CSV_PATH)
meta["patient_id"] = meta["patient_id"].astype(str).str.zfill(3)
print(f"Loaded metadata with {len(meta)} rows and columns: {list(meta.columns)}")

# --- Process each patient ---
for _, row in meta.iterrows():
    patient_id = row["patient_id"]
    num_windows = int(row["n_windows"])
    start_indices_raw = row["start_indices"]

    # Parse the stringified list safely
    try:
        start_indices = ast.literal_eval(start_indices_raw)
    except Exception as e:
        print(f"Could not parse start_indices for {patient_id}: {e}")
        continue

    # Validate
    if not isinstance(start_indices, (list, tuple)):
        print(f"start_indices for {patient_id} is not a list, skipping.")
        continue

    print(f"\n=== Processing patient {patient_id} ({len(start_indices)} windows) ===")

    # --- Load ECG file ---
    npz_path = os.path.join(PREPROCESSED_DIR, f"{patient_id}_preprocessed.npz")
    if not os.path.exists(npz_path):
        print(f"ECG file not found for patient {patient_id}")
        continue

    data = np.load(npz_path)
    key = list(data.keys())[0]  # assume ECG under first key
    ecg = data[key]
    total_len = len(ecg)
    print(f"  ECG length: {total_len} samples")

    # --- Create output directory ---
    patient_out_dir = os.path.join(SEGMENTS_DIR, patient_id)
    os.makedirs(patient_out_dir, exist_ok=True)

    # --- Segment and Save ---
    for i, start_idx in tqdm(enumerate(start_indices), total=len(start_indices),
                             desc=f"Segmenting {patient_id}", ncols=80):
        start_idx = int(start_idx)
        end_idx = start_idx + WINDOW_LEN
        if end_idx <= total_len:
            segment = ecg[start_idx:end_idx].astype(np.float32)
            out_path = os.path.join(patient_out_dir, f"{patient_id}_window{i:04d}.npy")
            np.save(out_path, segment)
        else:
            print(f"  Skipping window {i}: end_idx {end_idx} > {total_len}")

    num_saved = len([f for f in os.listdir(patient_out_dir) if f.endswith(".npy")])
    print(f"✅ Done: saved {num_saved}/{num_windows} windows for patient {patient_id}")

print("\nAll patients processed successfully!")
