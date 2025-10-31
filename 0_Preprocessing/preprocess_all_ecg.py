#!/usr/bin/env python3
"""
preprocess_all_ecgs.py

Batch-process SHDB-AF ambulatory ECGs by calling preprocess_ecg()
from preprocess_single_ecg.py. Discovers records by scanning *.hea
files and logs detailed metadata (including processing times).
"""

import os
import re
import csv
import time
import argparse
from tqdm import tqdm
from datetime import datetime
from preprocess_single_ecg import preprocess_ecg

DIGITS_RE = re.compile(r"^\d+$")


def discover_records(data_dir):
    """Return sorted list of record prefixes by scanning for *.hea files."""
    records = [f[:-4] for f in os.listdir(data_dir) if f.endswith(".hea")]
    numeric_records = [r for r in records if DIGITS_RE.match(r)]
    numeric_records.sort(key=lambda x: int(x))
    return numeric_records


def init_csv_log(csv_path):
    """Initialize CSV file with headers if not already present."""
    headers = [
        "timestamp",
        "record_id",
        "sampling_rate_hz",
        "signal_length",
        "num_channels",
        "duration_sec",
        "output_file",
        "output_size_mb",
        "load_time_s",
        "filter_time_s",
        "baseline_time_s",
        "normalize_time_s",
        "save_time_s",
        "total_time_s",
        "status",
        "error_msg",
    ]
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(headers)


def log_metadata(csv_path, meta):
    """Append a row of metadata to CSV."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            meta.get("record_id", ""),
            meta.get("sampling_rate_hz", ""),
            meta.get("signal_length", ""),
            meta.get("num_channels", ""),
            meta.get("duration_sec", ""),
            meta.get("output_file", ""),
            meta.get("output_size_mb", ""),
            meta.get("load_time_s", ""),
            meta.get("filter_time_s", ""),
            meta.get("baseline_time_s", ""),
            meta.get("normalize_time_s", ""),
            meta.get("save_time_s", ""),
            meta.get("total_time_s", ""),
            meta.get("status", ""),
            meta.get("error_msg", ""),
        ])


def main():
    parser = argparse.ArgumentParser(description="Batch preprocess SHDB-AF ECGs.")
    parser.add_argument(
        "--base_path",
        type=str,
        default="../../../../local3/sswee/data/shdbaf/physionet.org/files/shdb-af",
        help="Base path to SHDB-AF dataset (containing '1.0.1' and 'preprocessed').",
    )
    parser.add_argument("--version", type=str, default="1.0.1",
                        help="Dataset version folder under base_path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optionally limit number of records processed.")
    parser.add_argument("--csv_name", type=str, default="preprocessing_metadata.csv",
                        help="Output CSV filename for metadata logging.")
    args = parser.parse_args()

    data_dir = os.path.join(args.base_path, args.version)
    csv_path = os.path.join(args.base_path, args.csv_name)
    init_csv_log(csv_path)

    records = discover_records(data_dir)
    if args.limit:
        records = records[: args.limit]
    print(f"Found {len(records)} patient records in {data_dir}.")

    start = time.time()
    processed, errors = 0, 0

    for rec in tqdm(records, desc="Preprocessing SHDB-AF ECGs", unit="record"):
        try:
            meta = preprocess_ecg(rec, args.base_path)
            processed += 1
        except Exception as e:
            errors += 1
            meta = {
                "record_id": rec,
                "status": "error",
                "error_msg": str(e),
            }
            print(f"[!] Error on record {rec}: {e}")
        finally:
            log_metadata(csv_path, meta)

    total_min = (time.time() - start) / 60.0
    print(f"\nCompleted preprocessing. "
          f"Processed: {processed}, Errors: {errors}, Total time: {total_min:.2f} min.")
    print(f"Metadata saved to: {csv_path}")


if __name__ == "__main__":
    main()
