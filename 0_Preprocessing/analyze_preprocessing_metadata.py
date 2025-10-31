#!/usr/bin/env python3
"""
analyze_preprocessing_metadata.py

Goal:
-----
Analyze and visualize preprocessing performance from preprocessing_metadata.csv.

Generates:
----------
1. Summary statistics for all timing and size metrics.
2. Distribution histograms for each timing metric.
3. Scatter plots:
      - Signal length vs Total time
      - Output size vs Total time
4. Optional: top 5 slowest and fastest records.

Outputs:
--------
plots/
  ├── hist_load_time_s.png
  ├── hist_filter_time_s.png
  ├── ...
  ├── scatter_signal_length_vs_total_time.png
  ├── scatter_output_size_vs_total_time.png
analysis_summary.txt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
CSV_PATH = "preprocessing_metadata.csv"
OUTPUT_DIR = "plots"
SUMMARY_FILE = "analysis_summary.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} records from {CSV_PATH}")

# Keep only successful ones
df = df[df["status"] == "success"].copy()

# Convert timing columns to numeric safely
timing_cols = [
    "load_time_s", "filter_time_s", "baseline_time_s", "normalize_time_s",
    "save_time_s", "total_time_s"
]
for col in timing_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------------------------------------------------
# Compute summary statistics
# -------------------------------------------------------------------------
summary = df[timing_cols + ["output_size_mb", "signal_length", "duration_sec"]].describe()
summary.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))

with open(SUMMARY_FILE, "w") as f:
    f.write("=== Preprocessing Summary Statistics ===\n\n")
    f.write(summary.to_string())
    f.write("\n\n=== Top 5 Slowest Records ===\n")
    f.write(df.nlargest(5, "total_time_s")[["record_id", "total_time_s", "signal_length", "output_size_mb"]].to_string(index=False))
    f.write("\n\n=== Top 5 Fastest Records ===\n")
    f.write(df.nsmallest(5, "total_time_s")[["record_id", "total_time_s", "signal_length", "output_size_mb"]].to_string(index=False))

print("Summary and top records written to analysis_summary.txt")

# -------------------------------------------------------------------------
# Plot timing distributions
# -------------------------------------------------------------------------
for col in timing_cols:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col].dropna(), bins=20, color="steelblue", edgecolor="black")
    plt.title(f"Distribution of {col}")
    plt.xlabel("Seconds")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{col}.png"))
    plt.close()

print("Saved timing distribution histograms.")

# -------------------------------------------------------------------------
# Scatter plots: relationships between total_time_s and other variables
# -------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(df["signal_length"], df["total_time_s"], alpha=0.7)
plt.xlabel("Signal length (samples)")
plt.ylabel("Total preprocessing time (s)")
plt.title("Signal Length vs Total Preprocessing Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_signal_length_vs_total_time.png"))
plt.close()

plt.figure(figsize=(6, 4))
plt.scatter(df["output_size_mb"], df["total_time_s"], alpha=0.7, color="orange")
plt.xlabel("Output file size (MB)")
plt.ylabel("Total preprocessing time (s)")
plt.title("File Size vs Total Preprocessing Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_output_size_vs_total_time.png"))
plt.close()

print("Saved scatter plots of total time relationships.")

# -------------------------------------------------------------------------
# Compute throughput (MB/s)
# -------------------------------------------------------------------------
df["throughput_mb_per_s"] = df["output_size_mb"] / df["total_time_s"]
avg_throughput = df["throughput_mb_per_s"].mean()
print(f"Average throughput: {avg_throughput:.3f} MB/s")

plt.figure(figsize=(6, 4))
plt.hist(df["throughput_mb_per_s"].dropna(), bins=20, color="green", edgecolor="black")
plt.title("Distribution of Throughput (MB/s)")
plt.xlabel("MB per second")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hist_throughput.png"))
plt.close()

print("Saved throughput histogram.")
print("All plots saved in:", OUTPUT_DIR)
