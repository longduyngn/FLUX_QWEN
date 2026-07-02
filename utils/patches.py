import sys
import os
from pathlib import Path

def patch_diffusers():
    """Patches Diffusers telemetry and allocator warmups to save memory and avoid bugs."""
    try:
        import diffusers
        print("Imported diffusers, applying patches...")
    except ImportError:
        print("Diffusers not installed, skipping patches.")
        return

    # Patch 1: Serialization telemetry bug in diffusers.models.modeling_utils
    try:
        import diffusers.models.modeling_utils as modeling_utils
        path = Path(modeling_utils.__file__)
        text = path.read_text(encoding="utf-8")
        old = 'user_agent["quant_config"] = json.dumps(quantization_config.to_dict(), sort_keys=True)'
        new = 'user_agent["quant_config"] = json.dumps((quantization_config or hf_quantizer.quantization_config).to_dict(), sort_keys=True)'
        
        if old in text:
            try:
                path.write_text(text.replace(old, new), encoding="utf-8")
                print("Patched quant telemetry successfully:", path)
            except Exception as e:
                print(f"Failed to write quant telemetry patch (likely permission issue): {e}")
        elif new in text:
            print("Quant telemetry patch already applied:", path)
        else:
            print("Quant telemetry line not found (Diffusers version might differ or be pre-patched):", path)
    except Exception as e:
        print(f"Could not patch modeling_utils telemetry: {e}")

    # Patch 2: Disable caching allocator warmup (prevents huge initial VRAM peak)
    try:
        import diffusers.models.model_loading_utils as model_loading_utils
        
        def _skip_caching_allocator_warmup(*args, **kwargs):
            print("Skipped Diffusers caching allocator warmup to conserve VRAM.")
            return None

        if hasattr(model_loading_utils, "_caching_allocator_warmup"):
            model_loading_utils._caching_allocator_warmup = _skip_caching_allocator_warmup
            print("Patched model_loading_utils._caching_allocator_warmup")
            
        import diffusers.models.modeling_utils as modeling_utils
        if hasattr(modeling_utils, "_caching_allocator_warmup"):
            modeling_utils._caching_allocator_warmup = _skip_caching_allocator_warmup
            print("Patched modeling_utils._caching_allocator_warmup")
            
    except Exception as e:
        print(f"Could not patch caching allocator warmup: {e}")
