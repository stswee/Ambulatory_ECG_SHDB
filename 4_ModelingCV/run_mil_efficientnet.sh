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
PROJECT_DIR="$HOME/Ambulatory_ECG_SHDB/4_ModelingCV"
LOG_DIR="$PROJECT_DIR/logs"
SCRIPT_NAME="train_mil_efficientnet.py"

# ================================================================
# Parameters
# ================================================================
EPOCHS=30
BATCH_SIZE=1
MAX_SEGMENTS=64
LR=1e-4

# ================================================================
# Setup
# ================================================================
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/mil_efficientnet_${TIMESTAMP}.log"

# ================================================================
# Check for existing tmux session
# ================================================================
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Attaching..."
  tmux attach -t "$SESSION_NAME"
  exit 0
fi

echo "Starting new tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME"

# ================================================================
# Commands inside tmux
# ================================================================
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate $ENV_NAME" C-m
tmux send-keys "cd $PROJECT_DIR" C-m

tmux send-keys "echo 'Environment: $ENV_NAME'" C-m
tmux send-keys "echo 'Project Dir: $PROJECT_DIR'" C-m
tmux send-keys "echo 'Script: $SCRIPT_NAME'" C-m
tmux send-keys "echo 'Params: epochs=$EPOCHS, batch_size=$BATCH_SIZE, max_segments=$MAX_SEGMENTS, lr=$LR'" C-m
tmux send-keys "echo 'Log file: $LOGFILE'" C-m

tmux send-keys "python $SCRIPT_NAME --epochs $EPOCHS --batch_size $BATCH_SIZE --max_segments $MAX_SEGMENTS --mixed_precision | tee $LOGFILE" C-m

# ================================================================
# Info for user
# ================================================================
echo "Training started inside tmux session '$SESSION_NAME'."
echo "   To attach:   tmux attach -t $SESSION_NAME"
echo "   To detach:   Ctrl+B then D"
echo "   To kill:     tmux kill-session -t $SESSION_NAME"
echo "Logs:        $LOGFILE"
