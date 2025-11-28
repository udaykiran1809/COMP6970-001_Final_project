import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import gc
import os

def run_inpainting_pipeline(
    binary_mask: Image.Image,
    color_control_mask: Image.Image,
    original_image: Image.Image,
    prompt: str,
    negative_prompt: str = "ugly, blurry, low quality, deformed, distorted, bad anatomy",
    cache_dir: str = None
):
    """
    Loads models and runs the dedicated ControlNet Inpainting pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. Use the correct models for this pipeline ---
    CONTROLNET_MODEL_NAME = "lllyasviel/control_v11p_sd15_seg"
    INPAINTING_MODEL_NAME = "runwayml/stable-diffusion-inpainting"

    print(f"Loading ControlNet and Inpainting models...")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_NAME, torch_dtype=torch.float16, cache_dir=cache_dir
    )
    
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        INPAINTING_MODEL_NAME,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=cache_dir
    ).to(device)
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    print("Models loaded.")

    # --- 2. Prepare all three required images ---
    target_size = (512, 512)
    original_image_resized = original_image.resize(target_size)
    binary_mask_resized = binary_mask.resize(target_size).convert("L")
    color_control_mask_resized = color_control_mask.resize(target_size)
    
    # --- 3. Run the specialized Inpainting Pipeline ---
    print("Running dedicated ControlNet Inpainting pipeline...")
    generator = torch.manual_seed(42)
    
    final_image = pipe(
        prompt=prompt,
        image=original_image_resized,
        mask_image=binary_mask_resized,
        control_image=color_control_mask_resized,
        negative_prompt=negative_prompt,
        num_inference_steps=30, # Inpainting can benefit from a few more steps
        guidance_scale=8.0,
        generator=generator
    ).images[0]
    
    # --- 4. CRITICAL: VRAM Cleanup ---
    print("Cleaning up GPU memory...")
    del pipe
    del controlnet
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_image