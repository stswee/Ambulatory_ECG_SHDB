#!/usr/bin/env python3
import os
import numpy as np

# ---------------------------------------------------------------
# Paths for the single file
# ---------------------------------------------------------------
BASE_DIR = "../../ECG_data"
IN_FILE  = os.path.join(BASE_DIR, "preprocessed_segments/020/020_window1380.npy")
OUT_DIR  = os.path.join(BASE_DIR, "fft_amplitude_phase_segments/020")
OUT_FILE = os.path.join(OUT_DIR, "020_window1380_fft.npy")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# FFT amplitude + phase
# ---------------------------------------------------------------
def compute_amp_phase(signal):
    """
    Returns array of shape (2, L):
      [0, :] = amplitude
      [1, :] = phase
    """
    if signal.ndim == 2:
        signal = signal.squeeze(0)

    fft_vals = np.fft.rfft(signal)
    amp = np.abs(fft_vals).astype(np.float32)
    phase = np.angle(fft_vals).astype(np.float32)

    return np.stack([amp, phase], axis=0)

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    print(f"Loading {IN_FILE}")
    arr = np.load(IN_FILE)

    print("Computing FFT amplitude + phase...")
    amp_phase = compute_amp_phase(arr)

    print(f"Saving to {OUT_FILE}")
    np.save(OUT_FILE, amp_phase)

    print("\nDone! File saved successfully.")

if __name__ == "__main__":
    main()
