#!/usr/bin/env python3
"""
compute_fft_amplitude.py
------------------------
Convert raw ECG segments into Fourier amplitude spectra.

Input:
    ../../ECG_data/preprocessed_segments/<pid>/<pid>_windowXXX.npy

Output:
    ../../ECG_data/fft_amplitude_segments/<pid>/<pid>_windowXXX_amp.npy

Notes:
- Only FFT amplitude is saved.
- Shape is preserved: if input = (1, N), output = (N//2,) or (N//2+1,) depending on even N.
"""

import os
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
BASE_DIR = "../../ECG_data"
INPUT_DIR = os.path.join(BASE_DIR, "preprocessed_segments")
OUTPUT_DIR = os.path.join(BASE_DIR, "fft_amplitude_segments")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# FFT Function
# ---------------------------------------------------------------
def compute_amplitude(signal):
    """
    signal: numpy array shape (1, N) or (N,)
    returns: amplitude spectrum (positive frequencies)
    """
    if signal.ndim == 2:
        signal = signal.squeeze(0)

    N = signal.shape[0]
    fft_vals = np.fft.rfft(signal)   # real-valued FFT → half spectrum
    amp = np.abs(fft_vals)
    return amp.astype(np.float32)

# ---------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------
def main():
    patient_ids = sorted(os.listdir(INPUT_DIR))

    print(f"Found {len(patient_ids)} patient folders.")

    for pid in tqdm(patient_ids, desc="Patients"):
        in_pid_dir = os.path.join(INPUT_DIR, pid)
        out_pid_dir = os.path.join(OUTPUT_DIR, pid)
        os.makedirs(out_pid_dir, exist_ok=True)

        # Find segment files
        seg_files = sorted([
            f for f in os.listdir(in_pid_dir)
            if f.endswith(".npy") and "window" in f
        ])

        for seg_file in seg_files:
            in_path = os.path.join(in_pid_dir, seg_file)
            out_name = seg_file.replace(".npy", "_amp.npy")
            out_path = os.path.join(out_pid_dir, out_name)

            # Skip if already exists
            if os.path.exists(out_path):
                continue

            try:
                arr = np.load(in_path)

                amp = compute_amplitude(arr)

                np.save(out_path, amp)

            except Exception as e:
                print(f"Error processing {in_path}: {e}")

    print("\n FFT amplitude conversion finished!")
    print(f"Saved to: {OUTPUT_DIR}")

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
