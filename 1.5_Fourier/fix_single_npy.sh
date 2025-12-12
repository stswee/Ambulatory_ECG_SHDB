#!/bin/bash

TARGET_FILE="../../ECG_data/preprocessed_segments/020/020_window1380.npy"

echo "Checking permissions for: $TARGET_FILE"
ls -l "$TARGET_FILE"

echo ""
echo "Applying safe file permissions (chmod 644)..."
chmod 644 "$TARGET_FILE"

echo "Done. New permissions:"
ls -l "$TARGET_FILE"
