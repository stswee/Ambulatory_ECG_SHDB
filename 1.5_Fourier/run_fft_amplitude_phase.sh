#!/bin/bash
# ================================================================
# run_fft_amplitude.sh
# ------------------------------------------------
# Runs Fourier amplitude + phase conversion inside a tmux session.
#
# Environment: shdb-af-analysis
# Script: compute_fft_amplitude_phase.py
# Logs stored in: logs/
# ================================================================

SESSION_NAME="fft_amplitude_phase"
ENV_NAME="shdb-af-analysis"
PROJECT_DIR="$HOME/Ambulatory_ECG_SHDB/1.5_Fourier"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/fft_amplitude_phase_${TIMESTAMP}.log"

# Change this as needed:
GPU_ID=3   # You have access to GPUs 3–7

SCRIPT_NAME="compute_fft_amplitude_phase.py"

# ---------------------------------------------------------------
# Check if tmux session already exists
# ---------------------------------------------------------------
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
  echo "Attaching to existing tmux session: $SESSION_NAME"
  tmux attach -t $SESSION_NAME
  exit 0
fi

echo "Starting new tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

# ---------------------------------------------------------------
# Send environment setup + run command
# ---------------------------------------------------------------
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate $ENV_NAME" C-m
tmux send-keys "cd $PROJECT_DIR" C-m

# Set GPU visibility (even though FFT is CPU-only, keeps logs consistent)
tmux send-keys "export CUDA_VISIBLE_DEVICES=$GPU_ID" C-m

tmux send-keys "echo 'Running FFT amplitude + phase conversion on GPU $GPU_ID... Log: $LOGFILE'" C-m
tmux send-keys "python $SCRIPT_NAME | tee $LOGFILE" C-m

echo "FFT amplitude + phase conversion started inside tmux session '$SESSION_NAME'."
echo "To attach:  tmux attach -t $SESSION_NAME"
echo "To detach:  Ctrl+B then D"
echo "Logs saved to: $LOGFILE"
