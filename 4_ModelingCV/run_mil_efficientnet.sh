#!/bin/bash
# ================================================================
# run_mil_efficientnet.sh
# ------------------------------------------------
# Launches MIL ECG training with EfficientNet-1D encoder.
#
# Environment: shdb-af-analysis
# Script: train_mil_efficientnet.py
# GPU: CUDA_VISIBLE_DEVICES=2
# ================================================================

SESSION_NAME="mil_efficientnet"
ENV_NAME="shdb-af-analysis"
PROJECT_DIR="$HOME/Ambulatory_ECG_SHDB/4_ModelingCV"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/mil_efficientnet_${TIMESTAMP}.log"

EPOCHS=10
BATCH_SIZE=1
LR=1e-4
SCRIPT_NAME="train_mil_efficientnet.py"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
  echo "Attaching to session: $SESSION_NAME"
  tmux attach -t $SESSION_NAME
  exit 0
fi

echo "Starting session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate $ENV_NAME" C-m
tmux send-keys "cd $PROJECT_DIR" C-m

tmux send-keys "export CUDA_VISIBLE_DEVICES=2" C-m
tmux send-keys "echo 'Launching EfficientNet MIL training on GPU 2... Logs at $LOGFILE'" C-m
tmux send-keys "python $SCRIPT_NAME --epochs $EPOCHS --batch_size $BATCH_SIZE --mixed_precision | tee $LOGFILE" C-m

echo "EfficientNet training started in tmux session '$SESSION_NAME' on GPU 2."
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Detach: Ctrl+B then D"
