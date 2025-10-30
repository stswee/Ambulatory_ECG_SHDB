#!/usr/bin/env python3
"""
preprocess_all_ecgs.py

Batch-process SHDB-AF ambulatory ECGs by calling preprocess_ecg()
from preprocess_single_ecg.py. Discovers records by scanning *.hea
files in the 1.0.1 directory and uses tqdm for progress.

Usage:
  python preprocess_all_ecgs.py
  python preprocess_all_ecgs.py --base_path /abs/path/to/shdb-af --limit 143
"""

import os
import re
import time
import argparse
from tqdm import tqdm

# If this file sits next to preprocess_single_ecg.py, the import will work.
# If not, adjust sys.path accordingly.
from preprocess_single_ecg import preprocess_ecg

DIGITS_RE = re.compile(r"^\d+$")  # matches "001", "143", etc.


def discover_records(data_dir):
    """
    Return sorted list of record prefixes by scanning for *.hea files
    in the given directory (e.g., '001.hea' -> '001').
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    records = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".hea"):
            stem = fname[:-4]  # drop ".hea"
            records.append(stem)

    # Keep only numeric stems like '001', '002', ... and sort numerically
    numeric_records = [r for r in records if DIGITS_RE.match(r)]
    numeric_records.sort(key=lambda x: int(x))

    return numeric_records


def main():
    parser = argparse.ArgumentParser(description="Batch preprocess SHDB-AF ECGs.")
    parser.add_argument(
        "--base_path",
        type=str,
        default="../../../../local3/sswee/data/shdbaf/physionet.org/files/shdb-af",
        help="Base path to SHDB-AF dataset (containing '1.0.1' and 'preprocessed')",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.1",
        help="Dataset version folder under base_path (default: 1.0.1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of records processed (e.g., 143).",
    )
    args = parser.parse_args()

    data_dir = os.path.join(args.base_path, args.version)
    records = discover_records(data_dir)

    if args.limit is not None:
        records = records[: args.limit]

    print(f"Found {len(records)} patient records in {data_dir}.")

    start = time.time()
    processed = 0
    errors = 0

    for rec in tqdm(records, desc="Preprocessing SHDB-AF ECGs", unit="record"):
        try:
            preprocess_ecg(rec, args.base_path)  # from preprocess_single_ecg.py
            processed += 1
        except Exception as e:
            errors += 1
            print(f"Error on record {rec}: {e}")

    total_min = (time.time() - start) / 60.0
    print(
        f"\nCompleted preprocessing. "
        f"Processed: {processed}, Errors: {errors}, Total time: {total_min:.2f} min."
    )


if __name__ == "__main__":
    main()
