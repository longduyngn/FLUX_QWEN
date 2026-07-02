import os
import gc
import json
import torch
from transformers import T5EncoderModel, BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline,
    DiffusionPipeline,
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig
)

def resolve_path(project_root, path_str):
    """Resolves relative paths dynamically based on project root."""
    if not path_str:
        return None
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(project_root, path_str))

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

def build_flux_pipeline(config_path_or_dict, mode="text2img"):
    """
    Builds FLUX pipeline based on configurations.
    Supports:
      - flux-schnell (quantized 4-bit loading for T5 & Transformer)
      - flux-klein (using DiffusionPipeline directly)
    """
    if isinstance(config_path_or_dict, dict):
        config = config_path_or_dict
    else:
        with open(config_path_or_dict, "r", encoding="utf-8") as f:
            config = json.load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_name = config.get("model_name", "flux-schnell")
    repo_id = config.get("repo_id", "black-forest-labs/FLUX.1-schnell")
    
    # Check for local path override
    local_path = resolve_path(project_root, config.get("model_path"))
    load_source = local_path if (local_path and os.path.exists(local_path)) else repo_id

    hf_token = get_hf_token()
    is_cuda = torch.cuda.is_available()
    dtype = torch.float16 if is_cuda else torch.float32

    print(f"--- Assembling FLUX Pipeline ({model_name.upper()}) ---")
    print(f"Loading from: {load_source}")

    if model_name == "flux-klein":
        # FLUX.2-klein-4B pipeline loading
        print("Loading FLUX.2-klein-4B using DiffusionPipeline...")
        pipe = DiffusionPipeline.from_pretrained(
            load_source,
            torch_dtype=dtype,
            device_map="balanced" if is_cuda else None,
            token=hf_token
        )
        return pipe

    # Default to flux-schnell (4-bit quantized)
    if is_cuda:
        print("Using 4-bit Quantization Config for T5 and Transformer...")
        max_memory = config.get("max_memory", {0: "10GiB", "cpu": "30GiB"})
        
        # Quantization configurations
        text_encoder_2_quant = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        transformer_quant = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print("Loading quantized T5 Text Encoder...")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            load_source,
            subfolder="text_encoder_2",
            quantization_config=text_encoder_2_quant,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            token=hf_token
        )

        print("Loading quantized Transformer...")
        from diffusers import FluxTransformer2DModel
        transformer = FluxTransformer2DModel.from_pretrained(
            load_source,
            subfolder="transformer",
            quantization_config=transformer_quant,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            token=hf_token
        )

        print("Assembling final Flux pipeline components...")
        pipeline_class = FluxPipeline if mode == "text2img" else FluxImg2ImgPipeline
        pipe = pipeline_class.from_pretrained(
            load_source,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.float16,
            device_map="balanced",
            token=hf_token
        )
    else:
        # CPU Fallback (no quantization to prevent bitsandbytes CPU errors)
        print("CUDA not available. Loading FLUX pipeline on CPU (FP32, non-quantized)...")
        pipeline_class = FluxPipeline if mode == "text2img" else FluxImg2ImgPipeline
        pipe = pipeline_class.from_pretrained(
            load_source,
            torch_dtype=torch.float32,
            token=hf_token
        )

    # Enable slicing and tiling to optimize memory
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    return pipe


def run_qwen_image_workflow(config_path_or_dict, mode="text2img", init_image=None):
    """
    Executes the two-stage text-offloaded multi-GPU Qwen-Image workflow.
    Returns:
      PIL.Image: The generated image.
    """
    import torch
    from diffusers import QwenImagePipeline, QwenImageImg2ImgPipeline, QwenImageTransformer2DModel
    from utils.memory import free_memory

    # Load configuration
    if isinstance(config_path_or_dict, dict):
        config = config_path_or_dict
    else:
        with open(config_path_or_dict, "r", encoding="utf-8") as f:
            config = json.load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_id = config.get("repo_id", "diffusers/qwen-image-nf4")
    
    # Check for local path override
    local_path = resolve_path(project_root, config.get("model_path"))
    load_source = local_path if (local_path and os.path.exists(local_path)) else repo_id

    hf_token = get_hf_token()
    is_cuda = torch.cuda.is_available()
    
    # Qwen-Image is numerically sensitive in fp16; use bfloat16 to avoid NaN latents
    dtype = torch.bfloat16 if is_cuda else torch.float32
    print(f"--- Executing Qwen-Image Two-Stage Workflow ({mode.upper()}) ---")
    print(f"Model Source: {load_source}")
    print(f"Precision Dtype: {dtype}")

    # Stage 1: Load text encoder/tokenizer and precompute prompt embeddings.
    print("\n[Qwen Stage 1/3] Loading Text Encoder & Tokenizer...")
    text_pipe = QwenImagePipeline.from_pretrained(
        load_source,
        transformer=None,
        vae=None,
        torch_dtype=dtype,
        device_map="cuda" if is_cuda else "cpu",
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    prompt = config.get("prompt", "")
    negative_prompt = config.get("negative_prompt", " ")
    edit_prompt = config.get("edit_prompt", "")
    max_seq_len = config.get("max_sequence_length", 256)

    @torch.no_grad()
    def encode_qwen_text_only(pipe, prompt_text, max_sequence_length=256):
        device = pipe._execution_device
        prompt_list = [prompt_text] if isinstance(prompt_text, str) else prompt_text
        template = pipe.prompt_template_encode
        drop_idx = pipe.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt_list]
        txt_tokens = pipe.tokenizer(
            txt,
            max_length=pipe.tokenizer_max_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = pipe.text_encoder.model(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]

        split_hidden_states = pipe._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:max_sequence_length + drop_idx] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_len = max(e.size(0) for e in split_hidden_states)

        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_len - u.size(0), u.size(1))])
            for u in split_hidden_states
        ])
        prompt_embeds_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_len - u.size(0))])
            for u in attn_mask_list
        ])
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        if prompt_embeds_mask.all():
            prompt_embeds_mask = None
        return prompt_embeds, prompt_embeds_mask

    print("Encoding prompt embeddings...")
    prompt_embeds, prompt_embeds_mask = encode_qwen_text_only(text_pipe, prompt, max_sequence_length=max_seq_len)
    neg_embeds, neg_embeds_mask = encode_qwen_text_only(text_pipe, negative_prompt, max_sequence_length=max_seq_len)
    
    edit_embeds = None
    edit_embeds_mask = None
    if mode == "img2img" and edit_prompt:
        print("Encoding edit prompt embeddings...")
        edit_embeds, edit_embeds_mask = encode_qwen_text_only(text_pipe, edit_prompt, max_sequence_length=max_seq_len)

    # Move embeddings to CPU and clean up text pipeline
    prompt_embeds = prompt_embeds.detach().cpu()
    prompt_embeds_mask = None if prompt_embeds_mask is None else prompt_embeds_mask.detach().cpu()
    neg_embeds = neg_embeds.detach().cpu()
    neg_embeds_mask = None if neg_embeds_mask is None else neg_embeds_mask.detach().cpu()
    if edit_embeds is not None:
        edit_embeds = edit_embeds.detach().cpu()
        edit_embeds_mask = None if edit_embeds_mask is None else edit_embeds_mask.detach().cpu()

    print("Unloading Text pipeline to reclaim memory...")
    del text_pipe
    free_memory()

    # Stage 2: Load the sharded quantized transformer
    print("\n[Qwen Stage 2/3] Loading Transformer Component...")
    if is_cuda:
        transformer_max_memory = config.get("transformer_max_memory", {0: "14GiB", 1: "14GiB"})
        print(f"Loading Qwen Transformer sharded with max_memory={transformer_max_memory}...")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            load_source,
            subfolder="transformer",
            torch_dtype=dtype,
            device_map="balanced",
            max_memory=transformer_max_memory,
            low_cpu_mem_usage=True,
            token=hf_token,
        )
    else:
        print("CUDA not available. Loading Qwen Transformer on CPU...")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            load_source,
            subfolder="transformer",
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True,
            token=hf_token,
        )

    # Stage 3: Build pipeline without text encoder/tokenizer and execute inference
    print("\n[Qwen Stage 3/3] Assembling Inference Pipeline...")
    pipeline_class = QwenImagePipeline if mode == "text2img" else QwenImageImg2ImgPipeline
    
    pipe = pipeline_class.from_pretrained(
        load_source,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    exec_device = pipe._execution_device
    print(f"Pipeline execution device: {exec_device}")

    # Move and setup VAE
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.to(device=exec_device, dtype=dtype)
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        print(f"VAE loaded onto: {exec_device}")

    # Prepare embeddings for pipeline execution device
    run_prompt_embeds = prompt_embeds.to(device=exec_device, dtype=dtype)
    run_neg_embeds = neg_embeds.to(device=exec_device, dtype=dtype)
    run_prompt_mask = None if prompt_embeds_mask is None else prompt_embeds_mask.to(device=exec_device)
    run_neg_mask = None if neg_embeds_mask is None else neg_embeds_mask.to(device=exec_device)

    if mode == "img2img" and edit_embeds is not None:
        run_prompt_embeds = edit_embeds.to(device=exec_device, dtype=dtype)
        run_prompt_mask = None if edit_embeds_mask is None else edit_embeds_mask.to(device=exec_device)

    # Run latent pass
    print("Executing latent generation pass...")
    seed = config.get("edit_seed" if mode == "img2img" else "seed", 42)
    generator = torch.Generator("cpu").manual_seed(seed)
    
    height = config.get("height", 512)
    width = config.get("width", 512)
    steps = config.get("num_inference_steps", 8)
    cfg_scale = config.get("edit_guidance_scale" if mode == "img2img" else "guidance_scale", 1.0)

    @torch.no_grad()
    def run_latent_generation():
        if mode == "img2img":
            if not init_image:
                raise ValueError("init_image must be provided for img2img mode")
            return pipe(
                prompt_embeds=run_prompt_embeds,
                prompt_embeds_mask=run_prompt_mask,
                negative_prompt_embeds=run_neg_embeds,
                negative_prompt_embeds_mask=run_neg_mask,
                image=init_image.resize((width, height)),
                strength=config.get("strength", 0.55),
                true_cfg_scale=cfg_scale,
                width=width,
                height=height,
                num_inference_steps=steps,
                max_sequence_length=max_seq_len,
                generator=generator,
                output_type="latent",
            ).images
        else:
            return pipe(
                prompt_embeds=run_prompt_embeds,
                prompt_embeds_mask=run_prompt_mask,
                negative_prompt_embeds=run_neg_embeds,
                negative_prompt_embeds_mask=run_neg_mask,
                true_cfg_scale=cfg_scale,
                width=width,
                height=height,
                num_inference_steps=steps,
                max_sequence_length=max_seq_len,
                generator=generator,
                output_type="latent",
            ).images

    packed_latents = run_latent_generation()

    # VAE Decode in FP32 to avoid NaNs
    print("Decoding latents using FP32 VAE...")
    @torch.no_grad()
    def decode_fp32(packed_latents, target_device):
        pipe.vae.to(device=target_device, dtype=torch.float32)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        latents = packed_latents.to(device=target_device, dtype=torch.float32)
        latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
        
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        
        latents = latents / latents_std + latents_mean
        decoded = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
        
        finite_ratio = torch.isfinite(decoded).float().mean().item()
        print(f"Decoded finite ratio on {target_device}: {finite_ratio:.6f}")
        if finite_ratio < 1.0:
            raise RuntimeError(f"VAE decode produced NaN/Inf on {target_device}")
            
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    try:
        image = decode_fp32(packed_latents, exec_device)
    except Exception as exc:
        print(f"FP32 VAE decode on CUDA device {exec_device} failed, falling back to CPU decode: {exc}")
        image = decode_fp32(packed_latents, "cpu")

    # Clean up and reclaim memory
    print("Cleaning up generation pipeline...")
    del pipe, transformer
    free_memory()

    return image
