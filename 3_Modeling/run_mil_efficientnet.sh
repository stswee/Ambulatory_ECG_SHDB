#!/bin/bash
# ================================================================
# run_mil_efficientnet.sh
# ------------------------------------------------
# Launches Multiple-Instance Learning (MIL) ECG training with
# EfficientNet-1D encoder inside a tmux session.
#
# Environment: shdb-af-analysis
# Script: train_mil_efficientnet.py
# ================================================================

SESSION_NAME="mil_efficientnet"
ENV_NAME="shdb-af-analysis"
PROJECT_DIR="$HOME/Ambulatory_ECG_SHDB/3_Modeling"
LOG_DIR="$PROJECT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/mil_efficientnet_${TIMESTAMP}.log"

# Training parameters
EPOCHS=30
BATCH_SIZE=1
MAX_SEGMENTS=64
LR=1e-4
SCRIPT_NAME="train_mil_efficientnet.py"

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
  echo "Attaching to existing tmux session: $SESSION_NAME"
  tmux attach -t $SESSION_NAME
  exit 0
fi

echo "Starting new tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

# Send setup and training commands
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate $ENV_NAME" C-m
tmux send-keys "cd $PROJECT_DIR" C-m

tmux send-keys "echo 'Launching EfficientNet MIL training... Logs at $LOGFILE'" C-m
tmux send-keys "python $SCRIPT_NAME --epochs $EPOCHS --batch_size $BATCH_SIZE --max_segments $MAX_SEGMENTS --mixed_precision | tee $LOGFILE" C-m

echo "Training started inside tmux session '$SESSION_NAME'."
echo "To attach:  tmux attach -t $SESSION_NAME"
echo "To detach:  press Ctrl+B then D"
echo "Logs saved to: $LOGFILE"
