#!/bin/bash
# Kaggle Setup and Run script for FLUX & Qwen Local Inference Pipeline

echo "=== 1. Cloning Repository ==="
git clone https://github.com/longduyngn/FLUX_QWEN.git
cd FLUX_QWEN

echo "=== 2. Installing Dependencies ==="
pip install -r requirements.txt

echo "=== 3. Hugging Face Login (For gated models) ==="
if [ -n "$1" ]; then
    echo "Logging into Hugging Face Hub..."
    python -c "from huggingface_hub import login; login(token='$1')"
elif [ -n "$HF_TOKEN" ]; then
    echo "Logging into Hugging Face Hub using HF_TOKEN environment variable..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
else
    echo "No Hugging Face token provided. Skipping login."
    echo "If FLUX download fails, run this script as: bash scripts/kaggle_run.sh <your_hf_token>"
fi

echo "=== 4. Downloading Model Weights ==="
python scripts/download.py

echo "=== 5. Running Inference (FLUX Text2Img) ==="
python infer.py --model flux --mode text2img

echo "=== Process Complete! Check outputs/images/ ==="
