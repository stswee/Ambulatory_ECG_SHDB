#!/bin/bash

SESSION_NAME="ecg_preprocessing"
PYTHON_SCRIPT="preprocess_all_ecg.py"

# Create a new detached tmux session and run the Python script
tmux new-session -d -s "$SESSION_NAME" "python3 $PYTHON_SCRIPT"

echo "Tmux session '$SESSION_NAME' created and running '$PYTHON_SCRIPT'."
echo "To attach: tmux attach -t $SESSION_NAME"