#!/bin/bash
# ================================================================
# run_hmil_resnet1d.sh
# ------------------------------------------------
# Launches Hierarchical MIL (HMIL) ECG training with ResNet1D encoder.
#
# Environment: shdb-af-analysis
# Script: train_mil_resnet1d_hmil_cv.py
# GPU: CUDA_VISIBLE_DEVICES=1
# ================================================================

SESSION_NAME="hmil_resnet1d"
ENV_NAME="shdb-af-analysis"

# NOTE: Updated to point to 7_ModelingHMIL
PROJECT_DIR="$HOME/Ambulatory_ECG_SHDB/7_ModelingHMIL"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/hmil_resnet1d_${TIMESTAMP}.log"

# Hyperparameters
EPOCHS=10
BATCH_SIZE=1
LR=1e-4
HMIL_GROUP_SIZE=16
HMIL_NUM_LEVELS=2
SCRIPT_NAME="train_mil_resnet_1d.py"

# ---------------------------------------------------------------
# Start or attach to tmux session
# ---------------------------------------------------------------
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
  echo "Attaching to existing session: $SESSION_NAME"
  tmux attach -t $SESSION_NAME
  exit 0
fi

echo "Starting new HMIL session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate $ENV_NAME" C-m
tmux send-keys "cd $PROJECT_DIR" C-m

tmux send-keys "export CUDA_VISIBLE_DEVICES=1" C-m
tmux send-keys "echo 'Launching HMIL ResNet1D training on GPU 1... Logs at $LOGFILE'" C-m

tmux send-keys "python $SCRIPT_NAME \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --hmil_group_size $HMIL_GROUP_SIZE \
  --hmil_num_levels $HMIL_NUM_LEVELS \
  --mixed_precision | tee $LOGFILE" C-m

echo "HMIL training started in tmux session '$SESSION_NAME' on GPU 1."
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Detach: Ctrl+B then D"
