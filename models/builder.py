import os
import json
import torch
from transformers import CLIPTokenizer, T5TokenizerFast, CLIPTextModel, T5EncoderModel, BitsAndBytesConfig
from diffusers import AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxImg2ImgPipeline

def resolve_path(project_root, relative_path):
    """Resolves relative paths dynamically based on project root."""
    if not relative_path:
        return None
    return os.path.abspath(os.path.join(project_root, relative_path))

def build_flux_pipeline(config_path, mode="text2img"):
    """
    Builds the FLUX pipeline from localized model folders.
    Applies 4-bit quantization and setups custom component loading.
    
    Args:
        config_path (str): Absolute or relative path to the flux_config.json file.
        mode (str): Execution mode, either 'text2img' or 'img2img'.
    """
    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Resolve absolute paths
    vae_path = resolve_path(project_root, config.get("vae_path"))
    transformer_path = resolve_path(project_root, config.get("transformer_path"))
    text_encoder_path = resolve_path(project_root, config.get("text_encoder_path"))
    text_encoder_2_path = resolve_path(project_root, config.get("text_encoder_2_path"))
    tokenizer_path = resolve_path(project_root, config.get("tokenizer_path"))
    tokenizer_2_path = resolve_path(project_root, config.get("tokenizer_2_path"))
    scheduler_path = resolve_path(project_root, config.get("scheduler_path"))
    
    print("--- Assembling FLUX Pipeline Components ---")
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s).")
    
    # Determine if we should use manual offloading (only on 1 GPU to fit under 8GB VRAM)
    is_manual_offload = (num_gpus == 1)
    
    # Load VAE and Scheduler
    print(f"Loading VAE from {vae_path}...")
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
    if is_manual_offload:
        vae = vae.cpu()
        
    print(f"Loading Scheduler from {scheduler_path}...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)
    
    print(f"Loading Tokenizers...")
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    tokenizer_2 = T5TokenizerFast.from_pretrained(tokenizer_2_path)
    
    # Load Text Encoders
    print(f"Loading CLIP Text Encoder from {text_encoder_path}...")
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=torch.float16)
    if is_manual_offload:
        text_encoder = text_encoder.cpu()
        
    if is_manual_offload:
        # Load T5 with CPU offload helper (directly on GPU 0 then strip hooks)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=False
        )
        from accelerate.hooks import remove_hook_from_module
        print(f"Loading T5 Text Encoder (4-bit Quantized) onto GPU 0...")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            text_encoder_2_path,
            quantization_config=bnb_config,
            device_map={"": 0}
        )
        print("Removing T5 accelerate hooks and offloading to CPU...")
        remove_hook_from_module(text_encoder_2, recurse=True)
        text_encoder_2 = text_encoder_2.cpu()
        import gc; gc.collect(); torch.cuda.empty_cache()
    else:
        # 2+ GPUs: load using standard accelerate auto mapping
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=False
        )
        print(f"Loading T5 Text Encoder (4-bit Quantized) with auto device map...")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            text_encoder_2_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
    # Load Transformer
    if is_manual_offload:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=False
        )
        from accelerate.hooks import remove_hook_from_module
        print(f"Loading Transformer (4-bit Quantized) onto GPU 0...")
        try:
            transformer = FluxTransformer2DModel.from_pretrained(
                transformer_path,
                quantization_config=bnb_config,
                device_map={"": 0}
            )
            print("Removing Transformer accelerate hooks and offloading to CPU...")
            remove_hook_from_module(transformer, recurse=True)
            transformer = transformer.cpu()
            import gc; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"Quantized transformer load failed: {e}. Falling back to float16 on CPU.")
            transformer = FluxTransformer2DModel.from_pretrained(
                transformer_path,
                torch_dtype=torch.float16
            ).cpu()
    else:
        # 2+ GPUs: load using standard accelerate auto mapping
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=False
        )
        print(f"Loading Transformer (4-bit Quantized) with auto device map...")
        try:
            transformer = FluxTransformer2DModel.from_pretrained(
                transformer_path,
                quantization_config=bnb_config,
                device_map="auto"
            )
        except Exception as e:
            print(f"Direct transformer quantization loading failed: {e}")
            print("Loading transformer in float16 as fallback...")
            transformer = FluxTransformer2DModel.from_pretrained(
                transformer_path,
                torch_dtype=torch.float16
            )
            
    print(f"Combining components into Flux Pipeline (Mode: {mode})...")
    pipeline_class = FluxPipeline if mode == "text2img" else FluxImg2ImgPipeline
    pipeline = pipeline_class(
        scheduler=scheduler,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        transformer=transformer
    )
    
    pipeline.is_manual_offload = is_manual_offload
    return pipeline

def build_qwen_pipeline(config_path):
    """
    Builds the Qwen VL pipeline from localized model folders.
    Applies 4-bit quantization.
    
    Args:
        config_path (str): Absolute or relative path to the qwen_config.json file.
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = resolve_path(project_root, config.get("model_path"))
    
    print("--- Assembling Qwen2-VL Pipeline Components ---")
    
    print(f"Loading Qwen Processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 4-bit Quantization Config for Qwen Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    print(f"Loading Qwen2-VL Model (4-bit Quantized) from {model_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Create a pipeline wrapper class for convenient usage in execution
    class Qwen2VLPipeline:
        def __init__(self, model, processor):
            self.model = model
            self.processor = processor
            
        def __call__(self, prompt, image_path=None, max_new_tokens=512, **kwargs):
            from PIL import Image
            
            content = []
            if image_path and os.path.exists(image_path):
                print(f"Processing image: {image_path}")
                img = Image.open(image_path).convert("RGB")
                content.append({"type": "image", "image": img})
            else:
                if image_path:
                    print(f"Warning: Image file not found: {image_path}")
            
            content.append({"type": "text", "text": prompt})
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Extract inputs
            inputs = self.processor(
                text=[text],
                images=[Image.open(image_path).convert("RGB")] if (image_path and os.path.exists(image_path)) else None,
                padding=True,
                return_tensors="pt"
            )
            
            # Move tensors to GPU if available
            inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            print("Generating response...")
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
            
    pipeline = Qwen2VLPipeline(model, processor)
    return pipeline

def run_flux_step_by_step(
    pipeline,
    prompt,
    negative_prompt="",
    num_inference_steps=4,
    guidance_scale=0.0,
    height=1024,
    width=1024,
    seed=None,
    init_image=None,
    mode="text2img",
    strength=0.8
):
    """
    Executes FLUX denoising process step-by-step manually, bypassing high-level black box calls.
    Supports both text2img and img2img.
    """
    import torch
    import numpy as np

    is_manual_offload = getattr(pipeline, "is_manual_offload", False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running Step-by-Step FLUX Denoising (Mode: {mode.upper()}, Manual Offload: {is_manual_offload}) ---")

    # Disable grad-free context
    with torch.no_grad():
        # 1. Encode prompt
        print("Stage 1/5: Encoding Prompt (CLIP & T5 Encoders)")
        max_sequence_length = 256 if num_inference_steps <= 4 else 512
        
        if is_manual_offload:
            print("  Moving CLIP and T5 to GPU for encoding...")
            pipeline.text_encoder = pipeline.text_encoder.to(device)
            pipeline.text_encoder_2 = pipeline.text_encoder_2.to(device)
            
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        print(f"  Prompt Embeds Shape: {prompt_embeds.shape}")
        print(f"  Pooled Embeds Shape: {pooled_prompt_embeds.shape}")

        if is_manual_offload:
            print("  Offloading CLIP and T5 to CPU RAM...")
            pipeline.text_encoder = pipeline.text_encoder.cpu()
            pipeline.text_encoder_2 = pipeline.text_encoder_2.cpu()
            import gc; gc.collect(); torch.cuda.empty_cache()

        # 2. Prepare latents & Scheduler setup
        print("Stage 2/5: Initializing Latent Variables & Scheduler")
        generator = None
        if seed is not None:
            generator = torch.Generator(device).manual_seed(seed) if torch.cuda.is_available() else torch.Generator().manual_seed(seed)

        batch_size = 1

        if is_manual_offload:
            pipeline.vae = pipeline.vae.to(device)

        if mode == "img2img" and init_image is not None:
            print("  Preparing image-guided latents for img2img...")
            processed_init = pipeline.image_processor.preprocess(init_image, height=height, width=width)
            processed_init = processed_init.to(device=device, dtype=prompt_embeds.dtype)
            
            if hasattr(pipeline, "get_timesteps"):
                timesteps, num_inference_steps = pipeline.get_timesteps(num_inference_steps, strength, device)
            else:
                pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = pipeline.scheduler.timesteps
                start_idx = int(num_inference_steps * (1 - strength))
                timesteps = timesteps[start_idx:]
                num_inference_steps = len(timesteps)
                
            t_start = timesteps[0]
            t_tensor = t_start.expand(batch_size) if isinstance(t_start, torch.Tensor) else torch.tensor([t_start], device=device).expand(batch_size)
            
            latents, latent_image_ids = pipeline.prepare_latents(
                image=processed_init,
                timestep=t_tensor,
                batch_size=batch_size,
                num_images_per_prompt=1,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator
            )
        else:
            # text2img mode
            pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = pipeline.scheduler.timesteps
            
            num_channels_latents = pipeline.transformer.config.in_channels
            latents, latent_image_ids = pipeline.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
                latents=None
            )

        if is_manual_offload:
            pipeline.vae = pipeline.vae.cpu()
            import gc; gc.collect(); torch.cuda.empty_cache()

        print(f"  Latents Shape: {latents.shape}")
        print(f"  Number of Denoising Steps: {num_inference_steps}")

        # 3. Denoising Loop
        print("Stage 3/5: Executing Step-by-Step Denoising Loop")
        if is_manual_offload:
            print("  Moving Transformer to GPU...")
            pipeline.transformer = pipeline.transformer.to(device)

        transformer_device = next(pipeline.transformer.parameters()).device
        
        guidance = None
        if hasattr(pipeline.transformer.config, "guidance_embeds") or hasattr(pipeline.transformer, "guidance"):
            guidance = torch.tensor([guidance_scale], device=transformer_device).expand(latents.shape[0])

        # Move execution tensors to the correct submodule device
        latents = latents.to(transformer_device)
        latent_image_ids = latent_image_ids.to(transformer_device)
        prompt_embeds = prompt_embeds.to(transformer_device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_device)
        text_ids = text_ids.to(transformer_device)

        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                t_tensor = t.expand(latents.shape[0]) if isinstance(t, torch.Tensor) else torch.tensor([t], device=transformer_device).expand(latents.shape[0])
                t_tensor = t_tensor.to(transformer_device)
                
                noise_pred = pipeline.transformer(
                    hidden_states=latents,
                    timestep=t_tensor / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                
                latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                print(f"  [Step {i+1}/{num_inference_steps}] Timestep: {t.item():.1f}")
                progress_bar.update()

        if is_manual_offload:
            print("  Offloading Transformer to CPU RAM...")
            pipeline.transformer = pipeline.transformer.cpu()
            import gc; gc.collect(); torch.cuda.empty_cache()

        # 4. Unpack latents
        print("Stage 4/5: Unpacking latent dimensions")
        latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)

        # 5. Decode latents to pixel space (VAE)
        print("Stage 5/5: Decoding image via VAE Decoder")
        if is_manual_offload:
            print("  Moving VAE to GPU...")
            pipeline.vae = pipeline.vae.to(device)

        vae_device = next(pipeline.vae.parameters()).device
        latents = latents.to(vae_device)

        latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        image_tensor = pipeline.vae.decode(latents, return_dict=False)[0]

        if is_manual_offload:
            print("  Offloading VAE to CPU RAM...")
            pipeline.vae = pipeline.vae.cpu()
            import gc; gc.collect(); torch.cuda.empty_cache()

        image = pipeline.image_processor.postprocess(image_tensor, output_type="pil")[0]
        print("Generation complete!")
        return image

