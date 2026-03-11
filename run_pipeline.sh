#!/bin/bash

# Exit on error
set -e

# Define directories relative to this script's location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

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
