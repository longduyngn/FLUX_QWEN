import os
import sys
from huggingface_hub import snapshot_download

def get_hf_token():
    """Retrieves Hugging Face token from environment, Colab secrets, or Kaggle secrets."""
    for name in ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"]:
        val = os.environ.get(name)
        if val:
            return val.strip()
    try:
        from google.colab import userdata
        for name in ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"]:
            try:
                val = userdata.get(name)
                if val:
                    return val.strip()
            except Exception:
                pass
    except ImportError:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for name in ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"]:
            try:
                val = client.get_secret(name)
                if val:
                    return val.strip()
            except Exception:
                pass
    except ImportError:
        pass
    return None

def download_models():
    # Dynamically find the project root 
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    models = {
        "FLUX.1-schnell": {
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "local_dir": os.path.join(project_root, "models", "safefolder", "flux-schnell")
        },
        "FLUX.2-klein-4B": {
            "repo_id": "black-forest-labs/FLUX.2-klein-4B",
            "local_dir": os.path.join(project_root, "models", "safefolder", "flux-klein")
        },
        "Qwen-Image-NF4": {
            "repo_id": "diffusers/qwen-image-nf4",
            "local_dir": os.path.join(project_root, "models", "safefolder", "qwen-image-nf4")
        }
    }
    
    hf_token = get_hf_token()
    if hf_token:
        print("HF Token detected and will be used for authentication.")
    else:
        print("No HF Token detected. Downloading publicly accessible weights only.")

    for name, info in models.items():
        repo_id = info["repo_id"]
        local_dir = info["local_dir"]
        
        print(f"\n--- Downloading {name} ({repo_id}) ---")
        print(f"Target directory: {local_dir}")
        
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
                ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.ot"],
                local_dir_use_symlinks=False,
                token=hf_token
            )
            print(f"{name} downloaded successfully!")
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            print("Note: If the model requires license agreement, please run 'huggingface-cli login' first or check your token.")

if __name__ == "__main__":
    download_models()
