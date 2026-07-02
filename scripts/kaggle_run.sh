#!/bin/bash
# Kaggle Setup and Run script for FLUX & Qwen Local Inference Pipeline

# 1. Clone Repository (if not inside it)
if [ ! -d "configs" ]; then
    echo "=== 1. Cloning Repository ==="
    git clone https://github.com/longduyngn/FLUX_QWEN.git
    cd FLUX_QWEN
fi

echo "=== 2. Installing Dependencies ==="
pip install -r requirements.txt

# 3. Handle Hugging Face Token for Gated Models
if [ -n "$1" ]; then
    export HF_TOKEN="$1"
    python -c "from huggingface_hub import login; login(token='$1')"
elif [ -n "$HF_TOKEN" ]; then
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
fi

# 4. Search for pre-hosted Kaggle models
FLUX_SCHNELL_PATH=""
QWEN_PATH=""

if [ -d "/kaggle/input" ]; then
    echo "=== Searching for models in Kaggle input ==="
    # Find directories matching the model patterns
    FLUX_SCHNELL_PATH=$(find /kaggle/input -type d -name "*flux-schnell*" | head -n 1)
    QWEN_PATH=$(find /kaggle/input -type d -name "*qwen-image-nf4*" | head -n 1)
fi

# 5. Run inference using hosted models (no downloader needed)
if [ -n "$FLUX_SCHNELL_PATH" ]; then
    echo "Found FLUX.1-schnell hosted at: $FLUX_SCHNELL_PATH"
    echo "=== Running FLUX Inference (Hosted weights) ==="
    python infer.py --model flux-schnell --mode text2img --model_path "$FLUX_SCHNELL_PATH"
else
    echo "FLUX.1-schnell not found in Kaggle inputs. Will download weights on the fly."
    echo "=== Running FLUX Inference (On-demand download) ==="
    python infer.py --model flux-schnell --mode text2img
fi

if [ -n "$QWEN_PATH" ]; then
    echo "Found Qwen-Image hosted at: $QWEN_PATH"
    echo "=== Running Qwen-Image Inference (Hosted weights) ==="
    python infer.py --model qwen-image --mode text2img --model_path "$QWEN_PATH"
else
    echo "Qwen-Image not found in Kaggle inputs. Will download weights on the fly."
    echo "=== Running Qwen-Image Inference (On-demand download) ==="
    python infer.py --model qwen-image --mode text2img
fi

echo "=== Process Complete! Check outputs/images/ ==="
