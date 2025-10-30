#!/usr/bin/env bash
# ==============================================================
# run_preprocessing.sh
# --------------------------------------------------------------
# Launch batch preprocessing in a persistent tmux session.
# It activates your conda env and runs the Python batch script.
#
# Usage:
#   chmod +x run_preprocessing.sh
#   ./run_preprocessing.sh
# ==============================================================

set -euo pipefail

SESSION="ecg_preprocessing"
ENV_NAME="shdb-af-gpu"
# Adjust this path if the script is elsewhere:
PYTHON_SCRIPT="0 Preprocessing/preprocess_all_ecgs.py"
LOG_FILE="preprocessing_log_$(date +%Y-%m-%d_%H-%M-%S).txt"

# Helper: activate conda in non-interactive tmux shell
conda_init='eval "$(conda shell.bash hook)"'

# Create session if not exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attaching..."
  tmux attach -t "$SESSION"
  exit 0
fi

echo "🚀 Starting preprocessing in tmux session '$SESSION'..."
tmux new-session -d -s "$SESSION" \
"cd '$(pwd)' && $conda_init && conda activate $ENV_NAME && \
python \"$PYTHON_SCRIPT\" --limit 143 | tee \"$LOG_FILE\""

echo "Tmux session '$SESSION' started."
echo "Logs: $LOG_FILE"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl+B then D"
echo "Kill:   tmux kill-session -t $SESSION"
