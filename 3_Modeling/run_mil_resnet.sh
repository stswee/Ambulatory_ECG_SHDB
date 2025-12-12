#!/bin/bash
# ================================================================
# run_mil_train.sh
# ------------------------------------------------
# Launches MIL-ResNet1D training inside a tmux session
# Environment: shdb-af-analysis
# Script: train_mil_resnet1d.py
# ================================================================

SESSION_NAME="mil_train"
ENV_NAME="shdb-af-analysis"
PROJECT_DIR="$HOME/Ambulatory_ECG_SHDB/3_Modeling"
LOG_DIR="$PROJECT_DIR/logs"

# Create logs directory if not exists
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/mil_train_${TIMESTAMP}.log"

# Training arguments (you can modify as needed)
EPOCHS=30
BATCH_SIZE=1
MAX_SEGMENTS=64
LR=1e-4
SCRIPT_NAME="train_mil_resnet1d.py"

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
  echo "Attaching to existing tmux session: $SESSION_NAME"
  tmux attach -t $SESSION_NAME
  exit 0
fi

echo "Starting new tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

# Send commands to tmux session
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate $ENV_NAME" C-m
tmux send-keys "cd $PROJECT_DIR" C-m

# Run the training script
tmux send-keys "echo 'Launching training... Logs at $LOGFILE'" C-m
tmux send-keys "python $SCRIPT_NAME --epochs $EPOCHS --batch_size $BATCH_SIZE --max_segments $MAX_SEGMENTS --mixed_precision | tee $LOGFILE" C-m

echo "Training started inside tmux session '$SESSION_NAME'."
echo "To attach:  tmux attach -t $SESSION_NAME"
echo "To detach:  press Ctrl+B then D"
echo "Logs will be saved to: $LOGFILE"
