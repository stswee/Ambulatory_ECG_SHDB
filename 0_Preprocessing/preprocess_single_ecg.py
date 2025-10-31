#!/usr/bin/env python3
"""
preprocess_single_ecg.py

Purpose
-------
Preprocess one ambulatory ECG recording from the SHDB-AF dataset:
  1. Bandpass filter (0.5–40 Hz)
  2. Baseline correction (median filter)
  3. Z-score normalization
  4. Save as float32 .npz file in the preprocessed folder

Returns a metadata dictionary including all processing times.
"""

import os
import time
import argparse
import numpy as np
import wfdb
import scipy.signal as sp


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40, order=4):
    """Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    b, a = sp.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return sp.filtfilt(b, a, signal)


def baseline_correction(signal, fs, window_sec=0.8):
    """Remove baseline wander using a median filter."""
    kernel = int(window_sec * fs // 2 * 2 + 1)  # ensure odd kernel size
    baseline = sp.medfilt(signal, kernel_size=kernel)
    return signal - baseline


def preprocess_ecg(record_name, base_path):
    """Load, preprocess, and save a single ECG recording, returning metadata."""
    record_path = os.path.join(base_path, "1.0.1", record_name)
    save_dir = os.path.join(base_path, "preprocessed")
    os.makedirs(save_dir, exist_ok=True)

    meta = {"record_id": record_name}

    # -----------------------------
    # Load ECG
    # -----------------------------
    print(f"Loading record {record_name}...")
    t0 = time.time()
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]
    fs = record.fs
    load_time = time.time() - t0
    meta["load_time_s"] = round(load_time, 3)

    print(f"  Sampling rate: {fs} Hz, Signal length: {len(signal)} samples")
    print(f"  Load time: {load_time:.3f} s")

    # -----------------------------
    # Preprocessing steps
    # -----------------------------
    # Bandpass filter
    t1 = time.time()
    filtered = bandpass_filter(signal, fs)
    filter_time = time.time() - t1
    meta["filter_time_s"] = round(filter_time, 3)
    print(f"  Bandpass filtering took {filter_time:.3f} s")

    # Baseline correction
    t2 = time.time()
    baseline_corrected = baseline_correction(filtered, fs)
    baseline_time = time.time() - t2
    meta["baseline_time_s"] = round(baseline_time, 3)
    print(f"  Baseline correction took {baseline_time:.3f} s")

    # Z-score normalization
    t3 = time.time()
    normalized = (baseline_corrected - np.mean(baseline_corrected)) / np.std(baseline_corrected)
    normalized32 = normalized.astype(np.float32)
    normalize_time = time.time() - t3
    meta["normalize_time_s"] = round(normalize_time, 3)
    print(f"  Z-score normalization took {normalize_time:.3f} s")

    # -----------------------------
    # Save preprocessed ECG
    # -----------------------------
    save_path = os.path.join(save_dir, f"{record_name}_preprocessed.npz")
    t4 = time.time()
    np.savez_compressed(
        save_path,
        signal=normalized32,
        fs=fs,
        record_name=record_name,
        n_samples=len(normalized32)
    )
    save_time = time.time() - t4
    meta["save_time_s"] = round(save_time, 3)

    size_mb = os.path.getsize(save_path) / 1e6
    total_time = load_time + filter_time + baseline_time + normalize_time + save_time

    print(f" Saved preprocessed ECG to: {save_path}")
    print(f"   → File size: {size_mb:.2f} MB (dtype={normalized32.dtype})")
    print(f"   → Save time: {save_time:.3f} s")

    # -----------------------------
    # Return metadata
    # -----------------------------
    meta.update({
        "sampling_rate_hz": fs,
        "signal_length": len(signal),
        "num_channels": 1,
        "duration_sec": round(len(signal) / fs, 2),
        "output_file": save_path,
        "output_size_mb": round(size_mb, 3),
        "total_time_s": round(total_time, 3),
        "status": "success",
        "error_msg": "",
    })
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess one SHDB-AF ECG record.")
    parser.add_argument("--record", type=str, required=True, help="Record name (e.g., 001)")
    parser.add_argument(
        "--base_path",
        type=str,
        default="../../../../local3/sswee/data/shdbaf/physionet.org/files/shdb-af",
        help="Base path to SHDB-AF dataset directory"
    )
    args = parser.parse_args()

    meta = preprocess_ecg(args.record, args.base_path)
    print("\nMetadata summary:", meta)
