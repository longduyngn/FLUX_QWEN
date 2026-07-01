import gc
import torch

def free_memory():
    """Executes garbage collection and empties the PyTorch CUDA cache to free up VRAM."""
    print("Freeing memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

def optimize_pipeline(pipeline):
    """Applies CPU offloading to the pipeline to keep memory footprint under 8GB VRAM (only on single GPU)."""
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        print(f"Detected {num_gpus} GPUs. Skipping CPU offload to run entirely in VRAM for maximum speed.")
        return

    if hasattr(pipeline, "enable_model_cpu_offload"):
        print("Applying enable_model_cpu_offload to the pipeline...")
        pipeline.enable_model_cpu_offload()
    elif hasattr(pipeline, "model") and hasattr(pipeline.model, "enable_model_cpu_offload"):
        # For wrappers or models that support offload via subcomponents
        print("Applying enable_model_cpu_offload to the model subcomponent...")
        pipeline.model.enable_model_cpu_offload()
    else:
        print("Pipeline or model does not support enable_model_cpu_offload directly.")
        # Fallback: if model has a `.to("cuda")` we could use it, or just rely on bitsandbytes / accelerate device_map
        if hasattr(pipeline, "device") and str(pipeline.device) != "cuda":
            try:
                pipeline.to("cuda")
                print("Moved pipeline to CUDA.")
            except Exception as e:
                print(f"Could not move pipeline to CUDA: {e}")
