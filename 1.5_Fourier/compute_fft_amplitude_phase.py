#!/usr/bin/env python3
"""
compute_fft_amplitude_phase.py
------------------------------
Convert raw ECG segments into Fourier amplitude + phase spectra.

Input:
    ../../ECG_data/preprocessed_segments/<pid>/<pid>_windowXXX.npy

Output:
    ../../ECG_data/fft_amplitude_phase_segments/<pid>/<pid>_windowXXX_fft.npy

Output shape:
    (2, L):
        channel 0 = amplitude
        channel 1 = phase
"""

import os
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
BASE_DIR = "../../ECG_data"
INPUT_DIR = os.path.join(BASE_DIR, "preprocessed_segments")
OUTPUT_DIR = os.path.join(BASE_DIR, "fft_amplitude_phase_segments")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# FFT Function (Amplitude + Phase)
# ---------------------------------------------------------------
def compute_amplitude_phase(signal):
    """
    signal: numpy array shape (1, N) or (N,)
    returns: array shape (2, L) where:
        [0, :] = amplitude
        [1, :] = phase
    """
    if signal.ndim == 2:
        signal = signal.squeeze(0)

    fft_vals = np.fft.rfft(signal)   # half-spectrum (positive frequencies)
    amp = np.abs(fft_vals).astype(np.float32)
    phase = np.angle(fft_vals).astype(np.float32)

    return np.stack([amp, phase], axis=0)  # shape (2, L)

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

        seg_files = sorted([
            f for f in os.listdir(in_pid_dir)
            if f.endswith(".npy") and "window" in f
        ])

        for seg_file in seg_files:
            in_path = os.path.join(in_pid_dir, seg_file)
            out_name = seg_file.replace(".npy", "_fft.npy")
            out_path = os.path.join(out_pid_dir, out_name)

            # Skip existing files
            if os.path.exists(out_path):
                continue

            try:
                arr = np.load(in_path)

                amp_phase = compute_amplitude_phase(arr)

                np.save(out_path, amp_phase)

            except Exception as e:
                print(f"Error processing {in_path}: {e}")

    print("\nFFT amplitude + phase conversion finished!")
    print(f"Saved to: {OUTPUT_DIR}")

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
