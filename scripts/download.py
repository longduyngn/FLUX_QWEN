import os
from huggingface_hub import snapshot_download

def download_models():
    # Dynamically find the project root 
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    flux_dir = os.path.join(project_root, "models", "safefolder", "flux-schnell")
    qwen_dir = os.path.join(project_root, "models", "safefolder", "qwen-image")
    
    print(f"Project root resolved to: {project_root}")
    print(f"FLUX download target: {flux_dir}")
    print(f"Qwen download target: {qwen_dir}")
    
    # 1. Download FLUX.1-schnell
    print("\n--- Downloading FLUX.1-schnell ---")
    try:
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-schnell",
            local_dir=flux_dir,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot"],
            local_dir_use_symlinks=False
        )
        print("FLUX.1-schnell downloaded successfully!")
    except Exception as e:
        print(f"Error downloading FLUX.1-schnell: {e}")
        print("Note: If the model requires license agreement, please run 'huggingface-cli login' first.")

    # 2. Download Qwen2-VL-2B-Instruct
    print("\n--- Downloading Qwen2-VL-2B-Instruct ---")
    try:
        snapshot_download(
            repo_id="Qwen/Qwen2-VL-2B-Instruct",
            local_dir=qwen_dir,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot"],
            local_dir_use_symlinks=False
        )
        print("Qwen2-VL-2B-Instruct downloaded successfully!")
    except Exception as e:
        print(f"Error downloading Qwen2-VL-2B-Instruct: {e}")

if __name__ == "__main__":
    download_models()
