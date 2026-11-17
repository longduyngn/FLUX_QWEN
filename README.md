# FLUX & Qwen-Image Inference on Limited GPUs

Run large text-to-image models — **FLUX.1-schnell**, **FLUX.2-klein-4B** and **Qwen-Image** — on free/limited GPUs
(e.g. Kaggle T4 ×2) by combining 4-bit quantization, multi-GPU model splitting and CPU offloading.
Supports text-to-image and image-to-image / editing.

## Memory techniques used

| Technique | Where |
|---|---|
| 4-bit **NF4** quantization (bitsandbytes, double quant) for the FLUX transformer and T5 text encoder | `models/builder.py` |
| Pre-quantized `diffusers/qwen-image-nf4` transformer, **split across 2 GPUs** (`device_map="balanced"`, per-GPU `max_memory`) | `models/builder.py`, `configs/qwen_config.json` |
| **Two-stage Qwen workflow**: encode the prompt, free the text encoder, then load the transformer | `run_qwen_image_workflow` |
| **CPU offload** (`enable_model_cpu_offload`) on single-GPU machines; skipped when ≥2 GPUs | `utils/memory.py` |
| **BF16** for Qwen-Image (FP16 produced NaN latents) | `models/builder.py` |
| VAE slicing + tiling; Diffusers caching-allocator warmup disabled to lower peak VRAM at load | `models/builder.py`, `utils/patches.py` |

## Project structure

```
configs/        flux_config.json, qwen_config.json  (prompts, steps, resolution, memory caps)
models/         builder.py — pipeline construction for each model
utils/          memory.py, patches.py, report.py
scripts/        download.py (HF weights), kaggle_run.sh, extract_to_markdown.py
infer.py        CLI entry point
```

## Usage

```bash
pip install -r requirements.txt
export HF_TOKEN=...            # needed for gated FLUX weights

# text-to-image
python infer.py --model flux-schnell --mode text2img
python infer.py --model qwen-image  --mode text2img --prompt "a wooden cabin beside a lake"

# image-to-image / edit
python infer.py --model qwen-image --mode img2img --image path/to/input.png --edit_prompt "make it winter"
```

Other flags: `--steps`, `--seed`, `--config`, `--model_path` (use local weights, e.g. `/kaggle/input/...`).

On Kaggle: `bash scripts/kaggle_run.sh` installs dependencies, finds hosted weights under `/kaggle/input`
(or downloads them) and runs both models.

Outputs go to `outputs/images/`, and a timing report per run goes to `outputs/reports/`.
