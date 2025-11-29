#!/bin/bash
# ================================================================
# run_mil_efficientnet_crossattn.sh
# ------------------------------------------------
# Launches Multiple-Instance Learning (MIL) ECG training with
# EfficientNet-1D encoders and Cross-Attention fusion (time -> freq)
# inside a tmux session.
#
# Environment: shdb-af-analysis
# Script: train_mil_efficientnet1d_cv_crossattn.py
# GPU: CUDA_VISIBLE_DEVICES=1
# ================================================================

SESSION_NAME="mil_efficientnet1d_crossattn"
ENV_NAME="shdb-af-analysis"
PROJECT_DIR="$HOME/Ambulatory_ECG_SHDB/6_ModelingFusionCV"
LOG_DIR="$PROJECT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/mil_efficientnet_crossattn_${TIMESTAMP}.log"

# Training parameters
EPOCHS=10
BATCH_SIZE=1
LR=1e-4
SCRIPT_NAME="train_mil_efficientnet_crossattn.py"

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

# Set GPU and launch training
tmux send-keys "export CUDA_VISIBLE_DEVICES=2" C-m
tmux send-keys "echo 'Launching Cross-Attention MIL EfficientNet training on GPU 2... Logs at $LOGFILE'" C-m
tmux send-keys "python $SCRIPT_NAME --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --mixed_precision | tee $LOGFILE" C-m

echo "Training started inside tmux session '$SESSION_NAME' (GPU 2)."
echo "To attach:  tmux attach -t $SESSION_NAME"
echo "To detach:  press Ctrl+B then D"
echo "Logs saved to: $LOGFILE"
