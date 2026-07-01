import os
import sys
import time
import argparse
import json
from datetime import datetime
from PIL import Image, ImageDraw

# Add project root to path for local imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.builder import build_flux_pipeline, build_qwen_pipeline
from utils.memory import free_memory, optimize_pipeline
from utils.report import ReportGenerator

def render_text_to_image(text, width=800, height=600):
    """Renders text output onto a PIL Image for storage when output is textual."""
    img = Image.new("RGB", (width, height), color=(20, 20, 25))
    draw = ImageDraw.Draw(img)
    
    # Draw simple title
    draw.text((20, 20), "QWEN MODEL RESPONSE:", fill=(0, 255, 128))
    
    # Draw response lines
    y = 60
    margin = 20
    line_height = 20
    
    # Basic wrapping
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        # Simple length-based estimation instead of requiring custom fonts
        if len(test_line) * 7 > (width - 2 * margin):
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
        
    for line in lines:
        if y > height - margin:
            break
        draw.text((margin, y), line, fill=(240, 240, 240))
        y += line_height
        
    return img

def main():
    parser = argparse.ArgumentParser(description="Inference Runner for FLUX and Qwen Models")
    parser.add_argument("--model", type=str, choices=["flux", "qwen"], default="flux", help="Model to run")
    parser.add_argument("--mode", type=str, choices=["text2img", "img2img", "vqa"], default="text2img", help="Inference mode")
    parser.add_argument("--config", type=str, default=None, help="Override path to config json")
    parser.add_argument("--prompt", type=str, default=None, help="Override prompt for generation")
    parser.add_argument("--image", type=str, default=None, help="Input image path (for img2img or VQA)")
    args = parser.parse_args()

    # 1. Load Configurations
    if args.config:
        config_path = os.path.abspath(args.config)
    else:
        config_filename = f"{args.model}_config.json"
        config_path = os.path.join(project_root, "configs", config_filename)

    print(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Apply command line overrides
    if args.prompt:
        config["prompt"] = args.prompt
    if args.image:
        config["image_path"] = args.image

    # 2. Build Pipeline
    print(f"\nInitializing model pipeline: {args.model.upper()}...")
    pipeline = None
    try:
        if args.model == "flux":
            pipeline = build_flux_pipeline(config_path, mode=args.mode)
        elif args.model == "qwen":
            pipeline = build_qwen_pipeline(config_path)
    except Exception as e:
        print(f"Failed to build pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Optimize memory
    print("\nOptimizing VRAM utilization...")
    optimize_pipeline(pipeline)
    free_memory()

    # 4. Run Inference
    print(f"\nExecuting inference for {args.model.upper()}...")
    start_time = time.time()
    
    output_image = None
    input_image_path = config.get("image_path")
    
    try:
        if args.model == "flux":
            import torch
            generator = None
            if "seed" in config and config["seed"] is not None:
                generator = torch.Generator("cuda").manual_seed(config["seed"]) if torch.cuda.is_available() else torch.Generator().manual_seed(config["seed"])
            
            if args.mode == "text2img":
                from models.builder import run_flux_step_by_step
                output_image = run_flux_step_by_step(
                    pipeline=pipeline,
                    prompt=config.get("prompt"),
                    negative_prompt=config.get("negative_prompt", ""),
                    num_inference_steps=config.get("num_inference_steps", 4),
                    guidance_scale=config.get("guidance_scale", 0.0),
                    height=config.get("height", 1024),
                    width=config.get("width", 1024),
                    seed=config.get("seed"),
                    init_image=None,
                    mode="text2img"
                )
                
            elif args.mode == "img2img":
                print(f"Running Image-to-Image generation with input image: {input_image_path}")
                if not input_image_path or not os.path.exists(input_image_path):
                    raise ValueError(f"Input image path is missing or invalid: {input_image_path}")
                
                init_image = Image.open(input_image_path).convert("RGB")
                
                from models.builder import run_flux_step_by_step
                output_image = run_flux_step_by_step(
                    pipeline=pipeline,
                    prompt=config.get("prompt"),
                    negative_prompt=config.get("negative_prompt", ""),
                    num_inference_steps=config.get("num_inference_steps", 4),
                    guidance_scale=config.get("guidance_scale", 0.0),
                    height=config.get("height", 1024),
                    width=config.get("width", 1024),
                    seed=config.get("seed"),
                    init_image=init_image,
                    mode="img2img",
                    strength=config.get("strength", 0.8)
                )
                
        elif args.model == "qwen":
            print("Running Visual Question Answering...")
            response_text = pipeline(
                prompt=config.get("prompt"),
                image_path=input_image_path,
                max_new_tokens=config.get("max_new_tokens", 512)
            )
            print(f"\nResponse: {response_text}\n")
            output_image = render_text_to_image(response_text)
            
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = time.time()
    
    # 5. Save output image
    images_dir = os.path.join(project_root, "outputs", "images")
    os.makedirs(images_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"output_{args.model}_{timestamp}.png"
    out_path = os.path.join(images_dir, out_filename)
    
    if output_image:
        output_image.save(out_path)
        print(f"Output saved to: {out_path}")
    
    # 6. Generate execution report
    ReportGenerator(
        start_time=start_time,
        end_time=end_time,
        prompt=config.get("prompt"),
        input_image=input_image_path,
        output_image=out_path
    )
    
    # 7. Final VRAM clean up
    print("\nPerforming final memory cleanup...")
    free_memory()
    print("Process complete.")

if __name__ == "__main__":
    main()
