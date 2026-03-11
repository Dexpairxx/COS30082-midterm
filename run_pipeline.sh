#!/bin/bash

# Auto-launch in tmux if not already inside a tmux session
if [ -z "$TMUX" ]; then
    echo "==================================================="
    echo "   Launching Pipeline in Background (tmux)        "
    echo "==================================================="
    echo "Session Name: cos30082_train"
    echo "To view progress later, type:"
    echo "    tmux attach -t cos30082_train"
    echo ""
    
    # Kill existing session if it exists to avoid conflicts
    tmux kill-session -t cos30082_train 2>/dev/null || true
    
    # Start a detached tmux session running this very script
    tmux new-session -d -s cos30082_train "bash \"$0\"; bash"
    echo "✅ Pipeline is now running in the background!"
    echo "You can safely close this terminal."
    exit 0
fi

# Exit on error (This part only runs INSIDE the tmux session)
set -e

# Define directories
BASE_DIR=$(pwd)
DATA_DIR="$BASE_DIR/data/train"
SAVE_DIR="$BASE_DIR/saved_models"

echo "==================================================="
echo "   COS30082 Image Classification Pipeline         "
echo "==================================================="

# 1. Check for data
if [ ! -d "$DATA_DIR" ]; then
    echo "[!] ERROR: Data directory '$DATA_DIR' not found."
    echo "    Please download the dataset from Google Drive manually."
    echo "    (Due to Google Drive limits, 'gdown' fails to download 10,000 files in a folder)."
    echo "    Extract the dataset and place the 'train' folder into: $(pwd)/data"
    echo "    The structure should be:  data/train/<class_name>/image.jpg"
    exit 1
fi

# 2. Setup Environment
echo "[*] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create save directory
mkdir -p "$SAVE_DIR"

# 3. Train Baseline CNN Model
echo ""
echo "==================================================="
echo "   Starting Training: Custom CNN Baseline          "
echo "==================================================="
# Adjust batch_size and epochs as needed based on server capability
python train.py \
    --data_dir "$DATA_DIR" \
    --model cnn \
    --epochs 30 \
    --batch_size 64 \
    --lr 0.001 \
    --save_dir "$SAVE_DIR"

# 4. Train Transfer Learning Model
echo ""
echo "==================================================="
echo "   Starting Training: ResNet18 Transfer Learning   "
echo "==================================================="
python train.py \
    --data_dir "$DATA_DIR" \
    --model resnet \
    --epochs 20 \
    --batch_size 64 \
    --lr 0.001 \
    --save_dir "$SAVE_DIR"

# 5. Evaluate and Generate Metrics
echo ""
echo "==================================================="
echo "   Generating Evaluation Metrics & Plots           "
echo "==================================================="
python evaluate.py \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR"

echo ""
echo "==================================================="
echo "   Pipeline Complete! Check '$SAVE_DIR'           "
echo "==================================================="
