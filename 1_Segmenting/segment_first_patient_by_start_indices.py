#!/usr/bin/env python3
"""
segment_first_patient_by_start_indices.py

Goal:
-----
Segment a single preprocessed ECG (patient 001) into 30-second windows
using the start indices in window_index_metadata.csv.

CSV format (columns):
--------------------------------------------------
| patient_id | ... | num_windows | start_indices |
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

FS = 200           # Hz
WINDOW_SEC = 30
WINDOW_LEN = FS * WINDOW_SEC  # 6000 samples
PATIENT_ID = "001"  # test with first patient only

# --- Load metadata ---
meta = pd.read_csv(CSV_PATH)
meta["patient_id"] = meta["patient_id"].astype(str).str.zfill(3)
print(f"Loaded metadata with {len(meta)} rows and columns: {list(meta.columns)}")

# --- Filter for patient 001 ---
row = meta.loc[meta["patient_id"] == PATIENT_ID]
if row.empty:
    raise ValueError(f"No metadata found for patient {PATIENT_ID}")

row = row.iloc[0]
num_windows = int(row.iloc[-2])      # penultimate column = num_windows
start_indices_raw = row.iloc[-1]     # last column = start_indices

# Parse the start indices list
try:
    start_indices = ast.literal_eval(start_indices_raw)
except Exception as e:
    raise ValueError(f"Failed to parse start_indices for {PATIENT_ID}: {e}")

if not isinstance(start_indices, (list, tuple)):
    raise ValueError(f"start_indices for {PATIENT_ID} is not a list")

print(f"Patient {PATIENT_ID}: {num_windows} windows, {len(start_indices)} start indices")

# --- Create output directory ---
patient_out_dir = os.path.join(SEGMENTS_DIR, PATIENT_ID)
os.makedirs(patient_out_dir, exist_ok=True)

# --- Load ECG ---
npz_path = os.path.join(PREPROCESSED_DIR, f"{PATIENT_ID}_preprocessed.npz")
if not os.path.exists(npz_path):
    raise FileNotFoundError(f"ECG file not found: {npz_path}")

print(f"Loading ECG from {npz_path}")
data = np.load(npz_path)
key = list(data.keys())[0]  # assumes the first key is the ECG
ecg = data[key]
total_len = len(ecg)
print(f"ECG shape: {ecg.shape}, dtype: {ecg.dtype}, total length: {total_len}")

# --- Segment and Save ---
for i, start_idx in tqdm(enumerate(start_indices), total=len(start_indices),
                         desc=f"Segmenting patient {PATIENT_ID}"):
    start_idx = int(start_idx)
    end_idx = start_idx + WINDOW_LEN
    if end_idx <= total_len:
        segment = ecg[start_idx:end_idx].astype(np.float32)
        out_path = os.path.join(patient_out_dir, f"{PATIENT_ID}_window{i:04d}.npy")
        np.save(out_path, segment)
    else:
        print(f"  Skipping window {i}: end_idx {end_idx} > signal length {total_len}")

print(f"Done! Saved {len(os.listdir(patient_out_dir))} segments to {patient_out_dir}")
