import os
import sys
import time
import argparse
import json
from datetime import datetime
from PIL import Image

# Add project root to path for local imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Set expandable segments to conserve CUDA memory allocations
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Apply diffusers library level patches before imports
from utils.patches import patch_diffusers
patch_diffusers()

import torch
from models.builder import build_flux_pipeline, run_qwen_image_workflow
from utils.memory import free_memory, optimize_pipeline
from utils.report import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="Inference Runner for FLUX and Qwen Models")
    parser.add_argument("--model", type=str, choices=["flux-schnell", "flux-klein", "qwen-image"], default="flux-schnell", help="Model to run")
    parser.add_argument("--mode", type=str, choices=["text2img", "img2img"], default="text2img", help="Inference mode")
    parser.add_argument("--config", type=str, default=None, help="Override path to config json")
    parser.add_argument("--prompt", type=str, default=None, help="Override prompt for generation")
    parser.add_argument("--edit_prompt", type=str, default=None, help="Override edit prompt for generation")
    parser.add_argument("--image", type=str, default=None, help="Input image path (for img2img)")
    parser.add_argument("--steps", type=int, default=None, help="Override number of inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Override generation seed")
    parser.add_argument("--model_path", type=str, default=None, help="Override path to local model weights (e.g. /kaggle/input/...)")
    args = parser.parse_args()

    # 1. Load Configurations
    if args.config:
        config_path = os.path.abspath(args.config)
    else:
        # Determine default config file
        config_prefix = "flux" if args.model.startswith("flux") else "qwen"
        config_filename = f"{config_prefix}_config.json"
        config_path = os.path.join(project_root, "configs", config_filename)

    print(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Merge command-line arguments to config
    overridden = False
    if args.prompt:
        config["prompt"] = args.prompt
        overridden = True
    if args.edit_prompt:
        config["edit_prompt"] = args.edit_prompt
        overridden = True
    if args.image:
        config["image_path"] = args.image
        overridden = True
    if args.steps:
        config["num_inference_steps"] = args.steps
        overridden = True
    if args.seed:
        config["seed"] = args.seed
        overridden = True

    if args.model_path:
        config["model_path"] = args.model_path

    # Save overrides back to the config file if any occurred
    if overridden:
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            print(f"Saved updated overrides back to configuration file: {config_path}")
        except Exception as e:
            print(f"Warning: Could not save configuration updates back to file: {e}")

    # Resolve input image path and download if missing in img2img mode
    input_image_path = config.get("image_path")
    if input_image_path:
        if not os.path.isabs(input_image_path):
            input_image_path = os.path.abspath(os.path.join(project_root, input_image_path))
        
        # If it's img2img mode and the file doesn't exist, try downloading a default one
        if args.mode == "img2img" and not os.path.exists(input_image_path):
            print(f"Input image not found at: {input_image_path}")
            print("Attempting to download default sample image from repository...")
            try:
                os.makedirs(os.path.dirname(input_image_path), exist_ok=True)
                import urllib.request
                default_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
                urllib.request.urlretrieve(default_url, input_image_path)
                print(f"Downloaded default image to {input_image_path}")
            except Exception as e:
                print(f"Failed to download default image: {e}")
                sys.exit(1)

    # 2. Run Inference Workflow
    print(f"\nExecuting inference workflow for {args.model.upper()}...")
    start_time = time.time()
    output_image = None

    try:
        if args.model.startswith("flux"):
            # FLUX models (Schnell or Klein)
            pipeline = build_flux_pipeline(config, mode=args.mode)
            
            # Memory optimization for FLUX on single GPU
            print("\nOptimizing VRAM utilization...")
            optimize_pipeline(pipeline)
            free_memory()

            # Set generator seed
            seed = config.get("seed", 42)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device).manual_seed(seed) if device == "cuda" else torch.Generator().manual_seed(seed)

            # Execution
            if args.mode == "text2img":
                print("Running FLUX Text-to-Image Generation...")
                output_image = pipeline(
                    prompt=config.get("prompt"),
                    height=config.get("height", 512),
                    width=config.get("width", 512),
                    num_inference_steps=config.get("num_inference_steps", 4),
                    guidance_scale=config.get("guidance_scale", 0.0),
                    max_sequence_length=config.get("max_sequence_length", 128),
                    generator=generator
                ).images[0]
            
            elif args.mode == "img2img":
                print(f"Running FLUX Image-to-Image Generation with input: {input_image_path}")
                if not input_image_path or not os.path.exists(input_image_path):
                    raise ValueError(f"Input image path is missing or invalid: {input_image_path}")
                
                init_image = Image.open(input_image_path).convert("RGB")
                prompt_to_use = config.get("edit_prompt") or config.get("prompt")
                
                output_image = pipeline(
                    prompt=prompt_to_use,
                    image=init_image.resize((config.get("width", 512), config.get("height", 512))),
                    strength=config.get("strength", 0.85),
                    num_inference_steps=config.get("num_inference_steps", 4),
                    guidance_scale=config.get("guidance_scale", 0.0),
                    max_sequence_length=config.get("max_sequence_length", 128),
                    generator=generator
                ).images[0]

        elif args.model == "qwen-image":
            # Qwen-Image model (quantized text-offloaded workflow)
            init_image = None
            if args.mode == "img2img":
                print(f"Loading input image for Qwen-Image Img2Img/Edit: {input_image_path}")
                if not input_image_path or not os.path.exists(input_image_path):
                    raise ValueError(f"Input image path is missing or invalid: {input_image_path}")
                init_image = Image.open(input_image_path).convert("RGB")

            output_image = run_qwen_image_workflow(config, mode=args.mode, init_image=init_image)

    except Exception as e:
        print(f"Error during inference execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = time.time()

    # 3. Save Output Image
    images_dir = os.path.join(project_root, "outputs", "images")
    os.makedirs(images_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"output_{args.model}_{timestamp}.png"
    out_path = os.path.join(images_dir, out_filename)
    
    if output_image:
        output_image.save(out_path)
        print(f"Output image saved successfully to: {out_path}")
    else:
        print("Warning: No output image was generated.")

    # 4. Generate Execution Report
    prompt_used = config.get("edit_prompt") if (args.mode == "img2img" and config.get("edit_prompt")) else config.get("prompt")
    ReportGenerator(
        start_time=start_time,
        end_time=end_time,
        prompt=prompt_used,
        input_image=input_image_path,
        output_image=out_path
    )

    # 5. Final memory clean up
    print("\nPerforming final memory cleanup...")
    free_memory()
    print("Process complete.")

if __name__ == "__main__":
    main()
